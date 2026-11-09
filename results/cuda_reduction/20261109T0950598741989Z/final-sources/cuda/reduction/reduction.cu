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

constexpr int Threads = 256;

// 기준 구현: thread 하나가 한 행을 순서대로 더합니다.
__global__ void row_sum_naive(const float* input, float* output, int h, int w, int pitch) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= h) return;

    float sum = 0;
    for (int col = 0; col < w; ++col) sum += input[row * pitch + col];
    output[row] = sum;
}

// block 하나가 한 행을 맡습니다. 인접 thread가 인접 열을 읽고,
// thread별 부분합을 shared memory에서 절반씩 줄입니다.
__global__ void row_sum_tree(const float* input, float* output, int h, int w, int pitch) {
    __shared__ float partial[Threads];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0;
    for (int col = tid; col < w; col += Threads) sum += input[row * pitch + col];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = Threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partial[0];
}

// 항상 32개 lane이 참여합니다. 유효 입력이 없는 lane도 0으로 참여하므로
// full mask가 안전합니다. Pascal에서 지원하는 sync shuffle을 사용합니다.
__device__ float warp_sum(float value) {
    for (int delta = 16; delta > 0; delta /= 2)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

// warp 안에서는 register shuffle, warp 사이에서는 8개 부분합만 공유합니다.
// tree와 입력 분배는 같고 block barrier는 한 번만 필요합니다.
__global__ void row_sum_shuffle(const float* input, float* output, int h, int w, int pitch) {
    __shared__ float warp_partial[Threads / 32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    float sum = 0;
    for (int col = tid; col < w; col += Threads) sum += input[row * pitch + col];
    sum = warp_sum(sum);
    if (lane == 0) warp_partial[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < Threads / 32 ? warp_partial[lane] : 0;
        sum = warp_sum(sum);
        if (lane == 0) output[row] = sum;
    }
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static float widen(uint16_t bits) {
    const uint32_t word = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, 4);
    return value;
}

template <typename T>
static void json_array(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::runtime_error("Usage: reduction fixture.bin report.json naive|tree|shuffle samples batch");
        const std::string variant = argv[3];
        const int samples = std::stoi(argv[4]), batch = std::stoi(argv[5]);
        if ((variant != "naive" && variant != "tree" && variant != "shuffle") ||
            samples < 5 || samples > 100 || batch < 1 || batch > 100)
            throw std::runtime_error("Invalid benchmark settings");

        // NPU producer와 같은 BF16 row-major 입력을 FP32로 정확히 확장합니다.
        std::ifstream file(argv[1], std::ios::binary);
        char magic[8];
        uint32_t dimensions[4]{};
        file.read(magic, 8);
        file.read(reinterpret_cast<char*>(dimensions), 16);
        const int h = dimensions[0], w = dimensions[1], hp = dimensions[2], wp = dimensions[3];
        if (!file || std::memcmp(magic, "AFRS0001", 8) || h < 1 || w < 1 ||
            hp != (h + 31) / 32 * 32 || wp != (w + 31) / 32 * 32 || hp > 16384 || wp > 16384)
            throw std::runtime_error("Invalid fixture header");
        std::vector<uint16_t> bits(size_t(hp) * wp);
        file.read(reinterpret_cast<char*>(bits.data()), std::streamsize(bits.size() * 2));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Invalid fixture length");
        std::vector<float> input(bits.size());
        std::transform(bits.begin(), bits.end(), input.begin(), widen);
        std::vector<double> reference(h, 0);
        bool exact = true;
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const float value = input[size_t(row) * wp + col];
                exact &= value == std::trunc(value);
                reference[row] += double(value);
            }
        }

        // Allocation, 초기화, CPU 검증은 kernel event 측정에서 제외합니다.
        float *device_input = nullptr, *device_output = nullptr, *pinned_input = nullptr, *pinned_output = nullptr;
        const size_t input_bytes = input.size() * sizeof(float), output_bytes = hp * sizeof(float);
        check(cudaMalloc(&device_input, input_bytes));
        check(cudaMalloc(&device_output, output_bytes));
        check(cudaMallocHost(&pinned_input, input_bytes));
        check(cudaMallocHost(&pinned_output, output_bytes));
        std::copy(input.begin(), input.end(), pinned_input);
        cudaStream_t stream;
        cudaEvent_t start, finish;
        check(cudaStreamCreate(&stream));
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&finish));
        check(cudaMemcpyAsync(device_input, pinned_input, input_bytes, cudaMemcpyHostToDevice, stream));

        auto launch = [&] {
            if (variant == "naive")
                row_sum_naive<<<(h + Threads - 1) / Threads, Threads, 0, stream>>>(device_input, device_output, h, w, wp);
            else if (variant == "tree")
                row_sum_tree<<<h, Threads, 0, stream>>>(device_input, device_output, h, w, wp);
            else
                row_sum_shuffle<<<h, Threads, 0, stream>>>(device_input, device_output, h, w, wp);
            check(cudaGetLastError());
        };

        double max_error = 0;
        int validations = 0;
        std::vector<float> first;
        auto validate = [&] {
            check(cudaMemcpyAsync(pinned_output, device_output, output_bytes, cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            for (int row = 0; row < hp; ++row) {
                const double got = pinned_output[row];
                const double expected = row < h ? reference[row] : 0;
                const double error = std::abs(got - expected);
                const double tolerance = exact ? 0 : 1e-4 + 1e-5 * std::abs(expected);
                if (!std::isfinite(got) || error > tolerance)
                    throw std::runtime_error("Output mismatch at row " + std::to_string(row));
                max_error = std::max(max_error, error);
            }
            const std::vector<float> current(pinned_output, pinned_output + hp);
            if (first.empty()) first = current;
            else if (current != first) throw std::runtime_error("Output changed across samples");
            ++validations;
        };
        auto sentinel = [&] {
            // 유효 행은 NaN sentinel로 초기화하고 padding은 0으로 유지합니다.
            check(cudaMemsetAsync(device_output, 0xff, h * sizeof(float), stream));
            check(cudaMemsetAsync(device_output + h, 0, (hp - h) * sizeof(float), stream));
        };

        sentinel();
        for (int i = 0; i < 20; ++i) launch();
        validate();
        std::vector<double> kernel_ms, pipeline_ms;
        for (int sample = 0; sample < samples; ++sample) {
            sentinel();
            check(cudaEventRecord(start, stream));
            for (int i = 0; i < batch; ++i) launch();
            check(cudaEventRecord(finish, stream));
            check(cudaEventSynchronize(finish));
            float elapsed = 0;
            check(cudaEventElapsedTime(&elapsed, start, finish));
            kernel_ms.push_back(double(elapsed) / batch);
            validate();

            sentinel();
            check(cudaStreamSynchronize(stream));
            const auto before = std::chrono::steady_clock::now();
            check(cudaMemcpyAsync(device_input, pinned_input, input_bytes, cudaMemcpyHostToDevice, stream));
            launch();
            check(cudaMemcpyAsync(pinned_output, device_output, output_bytes, cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            pipeline_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count());
            validate();
        }

        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        std::ofstream report(argv[2]);
        report << std::setprecision(14) << "{\"passed\":true,\"variant\":\"" << variant
               << "\",\"gpu\":\"" << device.name << "\",\"shape_hw\":[" << h << ',' << w
               << "],\"input_storage\":\"FP32 widened from BF16\",\"accumulation\":\"FP32\",\"threads\":" << Threads
               << ",\"shared_bytes\":" << (variant == "tree" ? Threads * 4 : variant == "shuffle" ? 32 : 0)
               << ",\"integer_exact\":" << (exact ? "true" : "false") << ",\"max_abs_error\":" << max_error
               << ",\"validated_outputs\":" << validations << ",\"warmups\":20,\"batch\":" << batch
               << ",\"h2d_bytes\":" << input_bytes << ",\"d2h_bytes\":" << output_bytes << ",\"row_sums\":";
        first.resize(h);
        json_array(report, first);
        report << ",\"kernel_event_batch_average_ms\":";
        json_array(report, kernel_ms);
        report << ",\"host_pipeline_ms\":";
        json_array(report, pipeline_ms);
        report << "}\n";
        if (!report) throw std::runtime_error("Report write failed");

        check(cudaEventDestroy(start));
        check(cudaEventDestroy(finish));
        check(cudaStreamDestroy(stream));
        check(cudaFree(device_input));
        check(cudaFree(device_output));
        check(cudaFreeHost(pinned_input));
        check(cudaFreeHost(pinned_output));
        std::cout << "PASS " << variant << ' ' << h << 'x' << w << " max_abs=" << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
