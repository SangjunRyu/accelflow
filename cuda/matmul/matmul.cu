#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// 한 thread가 출력 한 원소를 계산하는 기준 구현입니다.
// B 접근은 같은 warp의 이웃 thread 사이에서 연속적이지만, 타일을 직접 재사용하지 않습니다.
__global__ void naive(const float* a, const float* b, float* c, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0;
    for (int inner = 0; inner < k; ++inner)
        sum = fmaf(a[row * k + inner], b[inner * n + col], sum);
    c[row * n + col] = sum;
}

// 16×16 출력 타일을 256개 thread가 맡습니다.
// 한 번 읽은 A/B 타일을 shared memory에 보관해 block 안에서 각각 16번 사용합니다.
// 경계 thread도 두 __syncthreads()에 참여해야 다음 K 타일이 덮어써지지 않습니다.
template <int Tile, bool Texture = false>
__global__ void tiled(const float* a, const float* b, float* c, int m, int k, int n,
                      cudaTextureObject_t ta, cudaTextureObject_t tb) {
    __shared__ float at[Tile][Tile];
    __shared__ float bt[Tile][Tile];
    const int row = blockIdx.y * Tile + threadIdx.y;
    const int col = blockIdx.x * Tile + threadIdx.x;
    float sum = 0;

    for (int base = 0; base < k; base += Tile) {
        const int ak = base + threadIdx.x, bk = base + threadIdx.y;
        // 같은 shared-memory 배치와 FMA 순서를 유지하고 입력 읽기 경로만 바꿉니다.
        if constexpr (Texture) {
            at[threadIdx.y][threadIdx.x] = row < m && ak < k ?
                tex2D<float>(ta, float(ak) + 0.5f, float(row) + 0.5f) : 0;
            bt[threadIdx.y][threadIdx.x] = bk < k && col < n ?
                tex2D<float>(tb, float(col) + 0.5f, float(bk) + 0.5f) : 0;
        } else {
            at[threadIdx.y][threadIdx.x] = row < m && ak < k ? a[row * k + ak] : 0;
            bt[threadIdx.y][threadIdx.x] = bk < k && col < n ? b[bk * n + col] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < Tile; ++inner)
            sum = fmaf(at[threadIdx.y][inner], bt[inner][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

static void check(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(expression) + " at line " + std::to_string(line) +
                                 ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(x) check((x), #x, __LINE__)

static uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0;
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

static float widen(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// NPU 보고서와 같은 tile/face 순서의 BF16 fingerprint를 계산합니다.
// CUDA의 실제 메모리 배치는 FP32 row-major이며, fingerprint 계산은 host에서만 합니다.
static uint64_t fingerprint(const std::vector<uint16_t>& values, int rows, int cols) {
    uint64_t hash = 14695981039346656037ULL;
    for (int tr = 0; tr < rows; tr += 32)
        for (int tc = 0; tc < cols; tc += 32)
            for (int fr = 0; fr < 2; ++fr)
                for (int fc = 0; fc < 2; ++fc)
                    for (int rr = 0; rr < 16; ++rr)
                        for (int cc = 0; cc < 16; ++cc) {
                            const auto bits = values[(tr + fr * 16 + rr) * cols + tc + fc * 16 + cc];
                            hash = (hash ^ uint8_t(bits)) * 1099511628211ULL;
                            hash = (hash ^ uint8_t(bits >> 8)) * 1099511628211ULL;
                        }
    return hash;
}

struct Fixture {
    uint32_t m = 0, k = 0, n = 0;
    int mp = 0, kp = 0, np = 0;
    std::vector<uint16_t> abits, bbits;
    std::vector<float> a, b;
};

static Fixture load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    char magic[8]{};
    input.read(magic, 8);
    if (std::memcmp(magic, "AFMM001", 8) != 0)
        throw std::runtime_error("Invalid fixture header");
    Fixture f;
    input.read(reinterpret_cast<char*>(&f.m), 4);
    input.read(reinterpret_cast<char*>(&f.k), 4);
    input.read(reinterpret_cast<char*>(&f.n), 4);
    if (!f.m || !f.k || !f.n || f.m > 4096 || f.k > 4096 || f.n > 4096)
        throw std::runtime_error("Unsupported dimensions");
    f.mp = int((f.m + 31) / 32 * 32);
    f.kp = int((f.k + 31) / 32 * 32);
    f.np = int((f.n + 31) / 32 * 32);
    f.abits.resize(size_t(f.mp) * f.kp);
    f.bbits.resize(size_t(f.kp) * f.np);
    input.read(reinterpret_cast<char*>(f.abits.data()), std::streamsize(f.abits.size() * 2));
    input.read(reinterpret_cast<char*>(f.bbits.data()), std::streamsize(f.bbits.size() * 2));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Truncated fixture or unexpected trailing data");
    for (auto x : f.abits) f.a.push_back(widen(x));
    for (auto x : f.bbits) f.b.push_back(widen(x));
    return f;
}

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

int main(int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::runtime_error("Usage: matmul fixture.bin report.json naive|tiled|texture_tiled warmups samples batch");
        const Fixture f = load(argv[1]);
        const std::string variant = argv[3];
        const int warmups = std::stoi(argv[4]), samples = std::stoi(argv[5]), batch = std::stoi(argv[6]);
        if ((variant != "naive" && variant != "tiled" && variant != "texture_tiled") || warmups < 1 || samples < 5 ||
            warmups > 100 || samples > 100 || batch < 1 || batch > 100)
            throw std::runtime_error("Invalid configuration");

        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        CUDA_CHECK(cudaSetDevice(0));
        int runtime = 0, driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));

        // CPU FP64 기준값은 NPU와 같은 양자화된 입력으로 계산합니다.
        // allocation, CPU 기준값, 입력 준비는 모든 측정 구간에서 제외합니다.
        const size_t acount = f.a.size(), bcount = f.b.size(), ccount = size_t(f.mp) * f.np;
        const auto integer = [](float x) { return std::isfinite(x) && x == std::trunc(x); };
        const bool exact = std::all_of(f.a.begin(), f.a.end(), integer) &&
                           std::all_of(f.b.begin(), f.b.end(), integer);
        std::vector<double> reference(ccount, 0);
        for (uint32_t row = 0; row < f.m; ++row)
            for (uint32_t col = 0; col < f.n; ++col)
                for (uint32_t inner = 0; inner < f.k; ++inner)
                    reference[size_t(row) * f.np + col] +=
                        double(f.a[size_t(row) * f.kp + inner]) * f.b[size_t(inner) * f.np + col];

        float *a = nullptr, *b = nullptr, *c = nullptr, *ha = nullptr, *hb = nullptr, *hc = nullptr;
        const bool texture = variant == "texture_tiled";
        cudaArray_t aa = nullptr, ba = nullptr;
        cudaTextureObject_t ta = 0, tb = 0;
        const auto resource_start = Clock::now();
        CUDA_CHECK(cudaMalloc(&c, ccount * 4));
        if (texture) {
            const auto channel = cudaCreateChannelDesc<float>();
            CUDA_CHECK(cudaMallocArray(&aa, &channel, f.kp, f.mp));
            CUDA_CHECK(cudaMallocArray(&ba, &channel, f.np, f.kp));
            auto create_texture = [](cudaArray_t array) {
                cudaResourceDesc resource{};
                resource.resType = cudaResourceTypeArray;
                resource.res.array.array = array;
                cudaTextureDesc descriptor{};
                descriptor.addressMode[0] = cudaAddressModeClamp;
                descriptor.addressMode[1] = cudaAddressModeClamp;
                descriptor.filterMode = cudaFilterModePoint;
                descriptor.readMode = cudaReadModeElementType;
                descriptor.normalizedCoords = 0;
                cudaTextureObject_t handle = 0;
                CUDA_CHECK(cudaCreateTextureObject(&handle, &resource, &descriptor, nullptr));
                return handle;
            };
            ta = create_texture(aa);
            tb = create_texture(ba);
        } else {
            CUDA_CHECK(cudaMalloc(&a, acount * 4));
            CUDA_CHECK(cudaMalloc(&b, bcount * 4));
        }
        // 배열/texture 또는 선형 버퍼를 만드는 host 비용입니다.
        const double resource_creation_host_ms = elapsed(resource_start);
        CUDA_CHECK(cudaMallocHost(&ha, acount * 4));
        CUDA_CHECK(cudaMallocHost(&hb, bcount * 4));
        CUDA_CHECK(cudaMallocHost(&hc, ccount * 4));
        std::copy(f.a.begin(), f.a.end(), ha);
        std::copy(f.b.begin(), f.b.end(), hb);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        auto launch = [&]() {
            const dim3 block(16, 16), grid((f.np + 15) / 16, (f.mp + 15) / 16);
            if (variant == "naive") naive<<<grid, block, 0, stream>>>(a, b, c, f.mp, f.kp, f.np);
            else if (texture) tiled<16, true><<<grid, block, 0, stream>>>(a, b, c, f.mp, f.kp, f.np, ta, tb);
            else tiled<16, false><<<grid, block, 0, stream>>>(a, b, c, f.mp, f.kp, f.np, 0, 0);
            CUDA_CHECK(cudaGetLastError());
        };
        auto write_inputs = [&]() {
            if (texture) {
                CUDA_CHECK(cudaMemcpy2DToArrayAsync(aa, 0, 0, ha, f.kp * 4, f.kp * 4, f.mp,
                                                   cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpy2DToArrayAsync(ba, 0, 0, hb, f.np * 4, f.np * 4, f.kp,
                                                   cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(a, ha, acount * 4, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(b, hb, bcount * 4, cudaMemcpyHostToDevice, stream));
            }
        };
        auto read_output = [&]() {
            CUDA_CHECK(cudaMemcpyAsync(hc, c, ccount * 4, cudaMemcpyDeviceToHost, stream));
        };
        double max_error = 0, bf16_max_error = 0;
        uint64_t output_fingerprint = 0;
        int validated_outputs = 0;
        std::vector<float> first;
        auto validate = [&]() {
            std::vector<uint16_t> rounded(ccount);
            for (int row = 0; row < f.mp; ++row)
                for (int col = 0; col < f.np; ++col) {
                    const size_t i = size_t(row) * f.np + col;
                    const double error = std::abs(double(hc[i]) - reference[i]);
                    const bool tail = row >= int(f.m) || col >= int(f.n);
                    const double tolerance = exact ? 0.0 : 1e-4 + 1e-5 * std::abs(reference[i]);
                    if (!std::isfinite(hc[i]) || error > tolerance ||
                        (tail && hc[i] != 0))
                        throw std::runtime_error("CUDA output mismatch at " + std::to_string(i));
                    max_error = std::max(max_error, error);
                    rounded[i] = bf16(hc[i]);
                    const double quantized_error = std::abs(double(widen(rounded[i])) - reference[i]);
                    if (quantized_error > 0.03 + 0.03 * std::abs(reference[i]))
                        throw std::runtime_error("BF16-rounded output outside common tolerance");
                    bf16_max_error = std::max(bf16_max_error, quantized_error);
                }
            if (first.empty()) first.assign(hc, hc + ccount);
            else if (!std::equal(first.begin(), first.end(), hc))
                throw std::runtime_error("Output changed between samples");
            output_fingerprint = fingerprint(rounded, f.mp, f.np);
            ++validated_outputs;
        };

        write_inputs();
        for (int i = 0; i < warmups; ++i) launch();
        read_output();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        validate();

        // 작은 kernel도 event 분해능의 영향을 줄이도록 batch회 묶어서 측정합니다.
        // 각 배치의 마지막 출력을 검사합니다. 단일 launch의 분포와는 구분합니다.
        std::vector<double> kernel_ms, host_resident_ms, h2d_ms, d2h_ms, host_pipeline_ms;
        for (int i = 0; i < samples; ++i) {
            CUDA_CHECK(cudaMemsetAsync(c, 0xff, ccount * 4, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int repeat = 0; repeat < batch; ++repeat) launch();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            kernel_ms.push_back(ms / batch);
            read_output();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            validate();

            // host에서 본 단일 launch와 완료 대기입니다. event kernel 시간과 구분합니다.
            CUDA_CHECK(cudaMemsetAsync(c, 0xff, ccount * 4, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto time = Clock::now();
            launch();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            host_resident_ms.push_back(elapsed(time));
            read_output();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            validate();

            CUDA_CHECK(cudaEventRecord(start, stream));
            write_inputs();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            h2d_ms.push_back(ms);

            CUDA_CHECK(cudaEventRecord(start, stream));
            read_output();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            d2h_ms.push_back(ms);

            CUDA_CHECK(cudaMemsetAsync(c, 0xff, ccount * 4, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            time = Clock::now();
            write_inputs();
            launch();
            read_output();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            host_pipeline_ms.push_back(elapsed(time));
            validate();
        }

        std::ofstream report(argv[2]);
        report << std::setprecision(12)
            << "{\n\"passed\":true,\"variant\":\"" << variant << "\",\"gpu\":\"" << device.name
            << "\",\"compute_capability\":[" << device.major << ',' << device.minor
            << "],\"sm_count\":" << device.multiProcessorCount << ",\"vram_bytes\":" << device.totalGlobalMem
            << ",\"runtime_version\":" << runtime << ",\"driver_api_version\":" << driver
            << ",\"shape_mkn\":[" << f.m << ',' << f.k << ',' << f.n
            << "],\"padded_mkn\":[" << f.mp << ',' << f.kp << ',' << f.np
            << "],\"storage_dtype\":\"fp32\",\"input_values\":\"widened BF16 fixture\",\"accumulation\":\"fp32 FMA\""
            << ",\"output_dtype\":\"fp32; BF16-rounded fingerprint on host\",\"block\":[16,16]"
            << ",\"shared_memory_bytes_per_block\":" << (variant == "naive" ? 0 : 2048)
            << ",\"resource_creation_host_ms\":" << resource_creation_host_ms
            << ",\"input_path\":\"" << (texture ? "cudaArray point tex2D" : "cudaMalloc linear") << "\""
            << ",\"texture_filter_is_point\":" << (texture ? "true" : "false")
            << ",\"warmups\":" << warmups << ",\"samples\":" << samples << ",\"launches_per_event_sample\":" << batch
            << ",\"validated_sample_outputs\":" << validated_outputs
            << ",\"integer_reference_requires_exact\":" << (exact ? "true" : "false")
            << ",\"max_abs_error\":" << max_error << ",\"bf16_rounded_max_abs_error\":" << bf16_max_error
            << ",\"input_fingerprint_a\":" << fingerprint(f.abits, f.mp, f.kp)
            << ",\"input_fingerprint_b\":" << fingerprint(f.bbits, f.kp, f.np)
            << ",\"bf16_output_fingerprint\":" << output_fingerprint
            << ",\"h2d_bytes\":" << (acount + bcount) * 4 << ",\"d2h_bytes\":" << ccount * 4
            << ",\"kernel_event_batch_average_ms\":";
        array(report, kernel_ms);
        report << ",\"host_resident_launch_finish_ms\":"; array(report, host_resident_ms);
        report << ",\"h2d_pinned_event_ms\":"; array(report, h2d_ms);
        report << ",\"d2h_pinned_event_ms\":"; array(report, d2h_ms);
        report << ",\"host_pipeline_h2d_kernel_d2h_ms\":"; array(report, host_pipeline_ms);
        report << "}\n";
        report.close();
        if (!report) throw std::runtime_error("Cannot write report");

        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
        if (texture) {
            CUDA_CHECK(cudaDestroyTextureObject(ta)); CUDA_CHECK(cudaDestroyTextureObject(tb));
            CUDA_CHECK(cudaFreeArray(aa)); CUDA_CHECK(cudaFreeArray(ba));
        } else {
            CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
        }
        CUDA_CHECK(cudaFree(c));
        CUDA_CHECK(cudaFreeHost(ha)); CUDA_CHECK(cudaFreeHost(hb)); CUDA_CHECK(cudaFreeHost(hc));
        std::cout << "CUDA_MATMUL_PASS " << variant << ' ' << f.m << ',' << f.k << ',' << f.n
                  << " max_abs=" << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
