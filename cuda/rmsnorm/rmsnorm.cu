#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int Threads = 256;

__device__ float warp_sum(float value) {
    for (int delta = 16; delta; delta /= 2)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

// 모든 warp가 참여해야 합니다. 첫 warp의 lane 0에 block 합을 반환합니다.
__device__ float block_sum(float value, float* partial) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (warp == 0) value = warp_sum(lane < Threads / 32 ? partial[lane] : 0.0f);
    return value;
}

// 기본 구현 stage 1: 제곱을 별도 global buffer에 저장합니다.
__global__ void square_input(const float* x, float* square, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) square[i] = __fmul_rn(x[i], x[i]);
}

// 기본 구현 stage 2: 각 행의 제곱합과 역 RMS를 계산합니다.
__global__ void inverse_rms(const float* square, float* scales, int w, int pitch, float epsilon) {
    __shared__ float partial[Threads / 32];
    const int row = blockIdx.x;
    float sum = 0;
    for (int col = threadIdx.x; col < w; col += Threads) sum += square[row * pitch + col];
    sum = block_sum(sum, partial);
    if (threadIdx.x == 0) scales[row] = rsqrtf(sum / float(w) + epsilon);
}

// 기본 구현 stage 3: 입력을 다시 읽고 행별 scale과 gamma를 적용합니다.
__global__ void normalize(const float* x, const float* gamma, const float* scales,
                          float* y, int h, int w, int pitch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * pitch) return;
    const int row = i / pitch, col = i % pitch;
    y[i] = col < w ? x[i] * scales[row] * gamma[col] : 0.0f;
}

// 한 block이 한 행을 맡아 입력을 shared memory에 보관합니다.
// 제곱 buffer와 scale buffer를 DRAM에 쓰지 않으며 입력도 두 번 읽지 않습니다.
__global__ void fused_rmsnorm(const float* x, const float* gamma, float* y,
                             int w, int pitch, float epsilon) {
    extern __shared__ float shared[];
    float* values = shared;
    float* partial = shared + w;
    __shared__ float scale;
    const int row = blockIdx.x;
    float sum = 0;
    for (int col = threadIdx.x; col < w; col += Threads) {
        const float value = x[row * pitch + col];
        values[col] = value;
        sum += __fmul_rn(value, value);
    }
    sum = block_sum(sum, partial);
    if (threadIdx.x == 0) scale = rsqrtf(sum / float(w) + epsilon);
    __syncthreads();
    for (int col = threadIdx.x; col < pitch; col += Threads)
        y[row * pitch + col] = col < w ? values[col] * scale * gamma[col] : 0.0f;
}

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static float widen(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, 4);
    return value;
}

static void write_array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("Usage: rmsnorm fixture.bin report.json unfused|fused samples batch");
        const std::string variant = argv[3];
        const int samples = std::stoi(argv[4]), batch = std::stoi(argv[5]);
        if ((variant != "unfused" && variant != "fused") || samples < 5 || samples > 100 || batch < 1 || batch > 100)
            throw std::runtime_error("Invalid settings");
        std::ifstream file(argv[1], std::ios::binary);
        char magic[8]; uint32_t dims[4]{}; float epsilon = 0;
        file.read(magic, 8);
        file.read(reinterpret_cast<char*>(dims), 16);
        file.read(reinterpret_cast<char*>(&epsilon), 4);
        const int h = dims[0], w = dims[1], hp = dims[2], wp = dims[3];
        if (!file || std::memcmp(magic, "AFRM0001", 8) || h < 1 || h > 2048 || w < 1 || w > 8193 ||
            hp != (h+31)/32*32 || wp != (w+31)/32*32 || !std::isfinite(epsilon) || epsilon <= 0)
            throw std::runtime_error("Invalid fixture header");
        const size_t count = size_t(hp) * wp;
        std::vector<uint16_t> bits(count + wp);
        file.read(reinterpret_cast<char*>(bits.data()), bits.size() * 2);
        if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Fixture length mismatch");
        std::vector<float> x(count), gamma(wp);
        for (size_t i = 0; i < count; ++i) x[i] = widen(bits[i]);
        for (int col = 0; col < wp; ++col) gamma[col] = widen(bits[count + col]);

        // 두 variant가 같은 FP64 CPU 기준값과 같은 허용오차를 사용합니다.
        std::vector<double> reference(count, 0);
        for (int row = 0; row < h; ++row) {
            double sum = 0;
            for (int col = 0; col < w; ++col) {
                const double value = x[row * wp + col];
                sum += value * value;
            }
            const double scale = 1 / std::sqrt(sum / w + double(epsilon));
            for (int col = 0; col < w; ++col)
                reference[row * wp + col] = double(x[row * wp + col]) * scale * gamma[col];
        }

        float *dx=nullptr, *dg=nullptr, *dy=nullptr, *dsquare=nullptr, *dscale=nullptr;
        float *px=nullptr, *pg=nullptr, *py=nullptr;
        const size_t bytes=count*4;
        check(cudaMalloc(&dx,bytes)); check(cudaMalloc(&dg,wp*4)); check(cudaMalloc(&dy,bytes));
        if (variant == "unfused") {
            check(cudaMalloc(&dsquare,bytes)); check(cudaMalloc(&dscale,hp*4));
        }
        check(cudaMallocHost(&px,bytes)); check(cudaMallocHost(&pg,wp*4)); check(cudaMallocHost(&py,bytes));
        std::copy(x.begin(),x.end(),px); std::copy(gamma.begin(),gamma.end(),pg);
        cudaStream_t stream; cudaEvent_t start,finish;
        check(cudaStreamCreate(&stream)); check(cudaEventCreate(&start)); check(cudaEventCreate(&finish));
        check(cudaMemcpyAsync(dx,px,bytes,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dg,pg,wp*4,cudaMemcpyHostToDevice,stream));

        auto launch = [&] {
            if (variant == "unfused") {
                square_input<<<(int(count)+Threads-1)/Threads,Threads,0,stream>>>(dx,dsquare,int(count));
                check(cudaGetLastError());
                inverse_rms<<<h,Threads,0,stream>>>(dsquare,dscale,w,wp,epsilon);
                check(cudaGetLastError());
                normalize<<<(h*wp+Threads-1)/Threads,Threads,0,stream>>>(dx,dg,dscale,dy,h,w,wp);
            } else {
                fused_rmsnorm<<<h,Threads,(w+Threads/32)*4,stream>>>(dx,dg,dy,w,wp,epsilon);
            }
            check(cudaGetLastError());
        };
        auto sentinel = [&] {
            check(cudaMemsetAsync(dy,0xff,h*wp*4,stream));
            check(cudaMemsetAsync(dy+h*wp,0,(hp-h)*wp*4,stream));
            if (variant == "unfused") {
                check(cudaMemsetAsync(dsquare,0xff,bytes,stream));
                check(cudaMemsetAsync(dscale,0xff,hp*4,stream));
            }
        };
        int validations=0;
        double max_error=0;
        std::vector<float> first;
        auto validate = [&] {
            check(cudaMemcpyAsync(py,dy,bytes,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            for (size_t i=0;i<count;++i) {
                const double got=py[i], expected=reference[i], error=std::abs(got-expected);
                const bool padding=int(i/wp)>=h || int(i%wp)>=w;
                if (!std::isfinite(got) || error>1e-4+1e-4*std::abs(expected) || (padding && got!=0))
                    throw std::runtime_error("RMSNorm mismatch at index "+std::to_string(i));
                max_error=std::max(max_error,error);
            }
            const std::vector<float> current(py,py+count);
            if (first.empty()) first=current;
            else if (current!=first) throw std::runtime_error("Repeated outputs differ");
            ++validations;
        };

        sentinel();
        for (int i=0;i<20;++i) launch();
        validate();
        std::vector<double> kernel_ms,pipeline_ms;
        for (int sample=0;sample<samples;++sample) {
            sentinel();
            check(cudaEventRecord(start,stream));
            for (int i=0;i<batch;++i) launch();
            check(cudaEventRecord(finish,stream)); check(cudaEventSynchronize(finish));
            float elapsed=0; check(cudaEventElapsedTime(&elapsed,start,finish));
            kernel_ms.push_back(double(elapsed)/batch);
            validate();

            sentinel(); check(cudaStreamSynchronize(stream));
            const auto before=std::chrono::steady_clock::now();
            check(cudaMemcpyAsync(dx,px,bytes,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(dg,pg,wp*4,cudaMemcpyHostToDevice,stream));
            launch();
            check(cudaMemcpyAsync(py,dy,bytes,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            pipeline_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count());
            validate();
        }
        cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device,0));
        std::ofstream report(argv[2]);
        report<<std::setprecision(14)<<"{\"passed\":true,\"variant\":\""<<variant<<"\",\"gpu\":\""<<device.name
              <<"\",\"shape_hw\":["<<h<<','<<w<<"],\"epsilon\":"<<epsilon<<",\"max_abs_error\":"<<max_error
              <<",\"validated_outputs\":"<<validations<<",\"padding_zero\":true,\"threads\":256,\"warmups\":20,\"batch\":"<<batch
              <<",\"kernel_launches_per_operator\":"<<(variant=="unfused"?3:1)
              <<",\"intermediate_global_bytes\":"<<(variant=="unfused"?bytes+hp*4:0)
              <<",\"shared_bytes_per_block\":"<<(variant=="unfused"?32:(w+Threads/32)*4+4)
              <<",\"h2d_bytes\":"<<bytes+wp*4<<",\"d2h_bytes\":"<<bytes<<",\"operator_event_batch_average_ms\":";
        write_array(report,kernel_ms);
        report<<",\"host_pipeline_ms\":"; write_array(report,pipeline_ms);
        // 작은 정확성 입력은 논리 출력을 저장해 NPU와 직접 비교합니다.
        if (h*w<=65536) {
            report<<",\"logical_output\":[";
            for(int row=0;row<h;++row) for(int col=0;col<w;++col) {
                if(row || col) report<<',';
                report<<first[row*wp+col];
            }
            report<<']';
        }
        report<<"}\n";
        if(!report) throw std::runtime_error("Report write failed");
        check(cudaEventDestroy(start)); check(cudaEventDestroy(finish)); check(cudaStreamDestroy(stream));
        check(cudaFree(dx)); check(cudaFree(dg)); check(cudaFree(dy));
        if(dsquare) check(cudaFree(dsquare)); if(dscale) check(cudaFree(dscale));
        check(cudaFreeHost(px)); check(cudaFreeHost(pg)); check(cudaFreeHost(py));
        std::cout<<"PASS RMSNorm "<<variant<<' '<<h<<'x'<<w<<" max_abs="<<max_error<<'\n';
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
