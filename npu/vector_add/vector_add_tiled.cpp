#include "common.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <cmath>
#include <random>

// Convert a row-major logical 32x32 tile into four 16x16 hardware faces.
static size_t tile_offset(size_t logical) {
    const size_t element = logical % 1024;
    const size_t row = element / 32, col = element % 32;
    return (logical / 1024) * 1024 + (row / 16 * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}

// BF16 Vector Add host: 입력 pack → 세 커널 파이프라인 구성 → 실행 → CPU 비교.
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("Usage: vector_add_tiled report.json [cb-depth: 1|2]");
        const uint32_t cb_depth = argc == 3 ? std::stoul(argv[2]) : 2;
        if (cb_depth != 1 && cb_depth != 2)
            throw std::runtime_error("CB depth must be 1 or 2");
        af::DeviceSession session;
        std::mt19937 rng(2026);
        nlohmann::json cases = nlohmann::json::array();

        // 각 사례를 1024원소 타일 단위로 올리고 남은 영역은 0으로 채웁니다.
        auto run = [&](const std::string& name,
                       const std::vector<float>& a,
                       const std::vector<float>& b,
                       bool exact) {
            if (a.empty() || a.size() != b.size())
                throw std::runtime_error("Invalid input length");
            const uint32_t n = a.size(), tiles = (n + 1023) / 1024, padded = tiles * 1024;
            std::vector<bfloat16> packed_a(padded, bfloat16(0.0f)), packed_b = packed_a;
            for (uint32_t i = 0; i < n; ++i) {
                packed_a[tile_offset(i)] = bfloat16(a[i]);
                packed_b[tile_offset(i)] = bfloat16(b[i]);
            }

            // 1. 입력/출력 DRAM을 할당하고 BF16 face 배치의 입력을 전송합니다.
            auto ad = af::buffer(session, padded * 2, 2048, af::BufferType::DRAM);
            auto bd = af::buffer(session, padded * 2, 2048, af::BufferType::DRAM);
            auto cd = af::buffer(session, padded * 2, 2048, af::BufferType::DRAM);
            auto& cq = session.device->mesh_command_queue();
            af::dist::EnqueueWriteMeshBuffer(cq, ad, packed_a, false);
            af::dist::EnqueueWriteMeshBuffer(cq, bd, packed_b, false);

            // 2. CB0/CB1은 입력, CB16은 출력입니다. depth는 각 CB에 보관할 타일 수입니다.
            auto program = af::CreateProgram();
            const af::CoreCoord core{0, 0};
            for (auto index : {af::CBIndex::c_0, af::CBIndex::c_1, af::CBIndex::c_16}) {
                af::CreateCircularBuffer(
                    program,
                    core,
                    af::CircularBufferConfig(cb_depth * 2048, {{index, af::DataFormat::Float16_b}})
                        .set_page_size(index, 2048));
            }

            // 3. 주소 해석용 compile args와 실행 shape용 runtime args를 구성합니다.
            std::vector<uint32_t> reader_args, writer_args;
            af::TensorAccessorArgs(*ad).append_to(reader_args);
            af::TensorAccessorArgs(*bd).append_to(reader_args);
            af::TensorAccessorArgs(*cd).append_to(writer_args);
            const std::string kernels = ACCELFLOW_KERNEL_DIR;
            af::DataMovementConfig reader_config{}, writer_config{};
            reader_config.processor = af::DataMovementProcessor::RISCV_0;
            reader_config.noc = af::NOC::RISCV_0_default;
            reader_config.compile_args = reader_args;
            writer_config.processor = af::DataMovementProcessor::RISCV_1;
            writer_config.noc = af::NOC::RISCV_1_default;
            writer_config.compile_args = writer_args;
            auto reader = af::CreateKernel(program, kernels + "/reader.cpp", core, reader_config);
            auto writer = af::CreateKernel(program, kernels + "/writer.cpp", core, writer_config);
            af::ComputeConfig compute_config{};
            compute_config.math_fidelity = af::MathFidelity::HiFi4;
            compute_config.fp32_dest_acc_en = false;
            compute_config.math_approx_mode = false;
            auto compute =
                af::CreateKernel(program, kernels + "/compute.cpp", core, compute_config);
            af::SetRuntimeArgs(program,
                               reader,
                               core,
                               {af::address32(ad->address()), af::address32(bd->address()), tiles});
            af::SetRuntimeArgs(program, writer, core, {af::address32(cd->address()), tiles});
            af::SetRuntimeArgs(program, compute, core, {tiles});

            // 4. 실행 완료 후 타일 배치의 출력을 읽습니다.
            af::execute(session, std::move(program));
            std::vector<bfloat16> actual;
            af::dist::EnqueueReadMeshBuffer(cq, actual, cd, true);
            if (actual.size() != padded)
                throw std::runtime_error("Output length mismatch");

            // 5. BF16 양자화 입력의 FP32 덧셈과 비교하고 padding이 0인지도 검사합니다.
            double max_abs = 0, sum_abs = 0, max_relative = 0;
            for (uint32_t i = 0; i < padded; ++i) {
                const size_t offset = tile_offset(i);
                // Compare against FP32 addition of the quantized BF16 inputs.
                const float expected = float(packed_a[offset]) + float(packed_b[offset]);
                const float got = float(actual[offset]);
                const double error = std::abs(double(got) - expected);
                const double tolerance = exact ? 0.0 : 0.008 + 0.008 * std::abs(expected);
                if (!std::isfinite(got) || error > tolerance || (i >= n && got != 0))
                    throw std::runtime_error(name + ": mismatch at " + std::to_string(i));
                if (i < n) {
                    max_abs = std::max(max_abs, error);
                    sum_abs += error;
                    max_relative =
                        std::max(max_relative, error / std::max(1e-6, double(std::abs(expected))));
                }
            }
            std::cout << "PASS tiled " << name << " n=" << n << " tiles=" << tiles
                      << " max_abs=" << max_abs << '\n';

            // 정확성과 소스 기반 payload 모델을 기록합니다. simulator 시간으로 성능을 주장하지
            // 않습니다.
            cases.push_back({{"name", name},
                             {"length", n},
                             {"padded_length", padded},
                             {"tile_count", tiles},
                             {"input_fingerprint_a", af::input_fingerprint(packed_a)},
                             {"input_fingerprint_b", af::input_fingerprint(packed_b)},
                             {"dram_buffer_observation", af::buffer_observation(ad)},
                             {"passed", true},
                             {"max_abs_error", max_abs},
                             {"mean_abs_error", sum_abs / n},
                             {"max_relative_error", max_relative},
                             {"relative_denominator_floor", 1e-6},
                             {"atol", exact ? 0.0 : 0.008},
                             {"rtol", exact ? 0.0 : 0.008},
                             {"logical_cb_capacity_bytes", cb_depth * 3 * 2048},
                             {"cb_count", 3},
                             {"cb_tiles_each", cb_depth},
                             {"kernel_count", 3},
                             {"core_count", 1},
                             {"dram_read_payload_bytes", tiles * 4096},
                             {"dram_write_payload_bytes", tiles * 2048},
                             {"padding_verified", true}});
        };

        // 같은 입력/seed로 CB depth 변경 전후를 비교할 수 있도록 사례 순서를 고정합니다.
        run("deterministic", {1, 2, 3, 4}, {5, 6, 7, 8}, true);
        run("zero", std::vector<float>(33, 0), std::vector<float>(33, 0), true);
        std::uniform_int_distribution<int> integers(-8, 8);
        for (uint32_t n : {1U, 31U, 32U, 33U, 1023U, 1024U, 1025U, 2048U, 2051U, 8193U, 65537U}) {
            std::vector<float> a(n), b(n);
            for (uint32_t i = 0; i < n; ++i) {
                a[i] = integers(rng);
                b[i] = integers(rng);
            }
            run("signed_integer_exact", a, b, true);
        }
        std::uniform_real_distribution<float> fractions(-1, 1);
        std::vector<float> a(2051), b(2051);
        for (uint32_t i = 0; i < a.size(); ++i) {
            a[i] = fractions(rng);
            b[i] = fractions(rng);
        }
        run("fractional_bf16", a, b, false);
        session.close();
        af::save(argv[1],
                 {{"variant", "tiled_reader_compute_writer"},
                  {"dtype", "bf16"},
                  {"seed", 2026},
                  {"reference", "FP32 addition of quantized BF16 inputs"},
                  {"cases", cases},
                  {"passed", true}});
        std::cout << "TILED_VECTOR_ADD_ALL_PASS cases=" << cases.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
