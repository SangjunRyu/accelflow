#include "../vector_add/common.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <random>
#include <string>

static uint32_t padded(uint32_t value) {
    return (value + 31) / 32 * 32;
}

// Row-major 좌표를 32×32 tile 및 16×16 face 안의 위치로 바꿉니다.
static size_t offset(uint32_t row, uint32_t col, uint32_t width) {
    const uint32_t r = row % 32, c = col % 32;
    return (row / 32 * (width / 32) + col / 32) * 1024 +
           (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16;
}

static std::vector<bfloat16> pack(const std::vector<bfloat16>& input,
                                 uint32_t height, uint32_t width) {
    std::vector<bfloat16> result(input.size());
    for (uint32_t row = 0; row < height; ++row)
        for (uint32_t col = 0; col < width; ++col)
            result[offset(row, col, width)] = input[row * width + col];
    if (result != tilize_nfaces(input, height, width))
        throw std::runtime_error("Input layout differs from official tilize_nfaces");
    return result;
}

// Host: BF16 입력 생성 → 3개 커널 구성 → 실행 → FP64 row-sum 기준과 비교.
int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5)
            throw std::runtime_error("Usage: reduction report.json depth:1|2 [repeats:1..5] [name:H,W]");
        const uint32_t depth = std::stoul(argv[2]);
        const uint32_t repeats = argc >= 4 ? std::stoul(argv[3]) : 3;
        const std::string selector = argc == 5 ? argv[4] : "";
        if ((depth != 1 && depth != 2) || repeats < 1 || repeats > 5)
            throw std::runtime_error("Invalid depth or repeat count");

        af::DeviceSession session;
        std::mt19937 rng(2026);
        nlohmann::json cases = nlohmann::json::array();
        auto run = [&](const std::string& name, uint32_t height, uint32_t width) {
            if (!selector.empty() && selector != name + ":" + std::to_string(height) + "," +
                                                     std::to_string(width)) return;
            const uint32_t hp = padded(height), wp = padded(width), ht = hp / 32, wt = wp / 32;
            const bool exact = name != "fractional";
            std::vector<bfloat16> input(hp * wp, bfloat16(0.0f));
            std::uniform_int_distribution<int> integer(-8, 8);
            std::uniform_real_distribution<float> fraction(-0.5f, 0.5f);
            std::vector<double> reference(hp, 0);

            // Padding은 0입니다. 기준값에는 BF16으로 양자화한 실제 입력을 사용합니다.
            for (uint32_t row = 0; row < height; ++row) {
                for (uint32_t col = 0; col < width; ++col) {
                    float value = float(integer(rng));
                    if (name == "fractional") value = fraction(rng);
                    if (name == "zero") value = 0;
                    if (name == "ones") value = 1;
                    if (name == "large_exact") value = 8;
                    if (name == "deterministic") value = float(row * width + col + 1);
                    if (name == "cancellation") {
                        value = col % 3 == 0 ? 4096.0f : col % 3 == 1 ? 1.0f : -4096.0f;
                        if (row % 2) value = -value;
                    }
                    input[row * wp + col] = bfloat16(value);
                    reference[row] += float(input[row * wp + col]);
                }
            }

            // 마지막 열 타일의 prefix 전체를 검사합니다. 마지막 prefix가 최종 행 합입니다.
            std::vector<double> prefix(ht * 1024, 0);
            for (uint32_t row = 0; row < height; ++row) {
                double running = 0;
                for (uint32_t col = 0; col < wp; ++col) {
                    running += float(input[row * wp + col]);
                    if (col >= wp - 32)
                        prefix[(row / 32) * 1024 + offset(col - (wp - 32), row % 32, 32)] = running;
                }
            }

            // Optional fixture export preserves the exact quantized input for CUDA.
            if (const char* directory = std::getenv("ACCELFLOW_REDUCTION_EXPORT_DIR")) {
                std::filesystem::create_directories(directory);
                const auto path = std::filesystem::path(directory) /
                    (name + "_" + std::to_string(height) + "x" + std::to_string(width) + ".bin");
                std::ofstream file(path, std::ios::binary);
                file.write("AFRS0001", 8);
                for (auto dimension : {height, width, hp, wp})
                    file.write(reinterpret_cast<const char*>(&dimension), 4);
                for (auto value : input) {
                    const uint16_t bits = uint16_t(std::bit_cast<uint32_t>(float(value)) >> 16);
                    file.write(reinterpret_cast<const char*>(&bits), 2);
                }
                if (!file) throw std::runtime_error("Fixture export failed");
            }

            const auto packed = pack(input, hp, wp);

            auto xd = af::buffer(session, hp * wp * 2, 2048, af::BufferType::DRAM);
            auto yd = af::buffer(session, ht * 4096, 4096, af::BufferType::DRAM);
            auto& queue = session.device->mesh_command_queue();
            af::dist::EnqueueWriteMeshBuffer(queue, xd, packed, false);

            // CB0=input BF16, CB16=output FP32 prefix tile.
            // 입력 폭이 커져도 이 CB 용량은 늘어나지 않습니다.
            auto program = af::CreateProgram();
            const af::CoreCoord core{0, 0};
            for (auto cb : {af::CBIndex::c_0, af::CBIndex::c_16}) {
                const uint32_t page = cb == af::CBIndex::c_16 ? 4096 : 2048;
                const uint32_t slots = depth;
                const auto format = cb == af::CBIndex::c_16 ? af::DataFormat::Float32 : af::DataFormat::Float16_b;
                af::CreateCircularBuffer(program, core,
                    af::CircularBufferConfig(page * slots, {{cb, format}}).set_page_size(cb, page));
            }

            af::DataMovementConfig reader_config{}, writer_config{};
            reader_config.processor = af::DataMovementProcessor::RISCV_0;
            reader_config.noc = af::NOC::RISCV_0_default;
            af::TensorAccessorArgs(*xd).append_to(reader_config.compile_args);
            writer_config.processor = af::DataMovementProcessor::RISCV_1;
            writer_config.noc = af::NOC::RISCV_1_default;
            af::TensorAccessorArgs(*yd).append_to(writer_config.compile_args);
            af::ComputeConfig math_config{};
            math_config.math_fidelity = af::MathFidelity::HiFi4;
            math_config.fp32_dest_acc_en = true;
            math_config.math_approx_mode = false;
            const std::string kernels = ACCELFLOW_KERNEL_DIR;
            auto reader = af::CreateKernel(program, kernels + "/reader.cpp", core, reader_config);
            auto compute = af::CreateKernel(program, kernels + "/compute.cpp", core, math_config);
            auto writer = af::CreateKernel(program, kernels + "/writer.cpp", core, writer_config);
            af::SetRuntimeArgs(program, reader, core,
                {af::address32(xd->address()), ht * wt});
            af::SetRuntimeArgs(program, compute, core, {ht, wt});
            af::SetRuntimeArgs(program, writer, core, {af::address32(yd->address()), ht});

            af::dist::MeshWorkload workload;
            workload.add_program(af::dist::MeshCoordinateRange(session.device->shape()), std::move(program));
            const std::vector<float> sentinel(ht * 1024, std::numeric_limits<float>::quiet_NaN());
            std::vector<float> first;
            double max_error = 0, sum_error = 0;
            std::vector<float> row_sums;
            for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
                af::dist::EnqueueWriteMeshBuffer(queue, yd, sentinel, false);
                af::dist::Finish(queue);
                af::dist::EnqueueMeshWorkload(queue, workload, false);
                af::dist::Finish(queue);
                std::vector<float> actual;
                af::dist::EnqueueReadMeshBuffer(queue, actual, yd, true);
                if (actual.size() != sentinel.size()) throw std::runtime_error("Output length mismatch");

                row_sums.clear();
                double this_sum_error = 0;
                for (uint32_t group = 0; group < ht; ++group) {
                    for (uint32_t scan_row = 0; scan_row < 32; ++scan_row) {
                        for (uint32_t lane = 0; lane < 32; ++lane) {
                            const uint32_t row = group * 32 + lane;
                            const size_t i = group * 1024 + offset(scan_row, lane, 32);
                            const double expected = prefix[i];
                            const double got = actual[i], error = std::abs(got - expected);
                            const double tolerance = exact ? 0.0 : 1e-4 + 1e-5 * std::abs(expected);
                            if (!std::isfinite(got) || error > tolerance || (row >= height && got != 0))
                                throw std::runtime_error(name + ": prefix mismatch at " + std::to_string(row) +
                                    "," + std::to_string(scan_row) + " expected=" + std::to_string(expected) +
                                    " actual=" + std::to_string(got));
                            max_error = std::max(max_error, error);
                            if (row < height && scan_row == 31) {
                                if (std::abs(expected - reference[row]) > 1e-12)
                                    throw std::runtime_error("CPU prefix and sum references differ");
                                this_sum_error += error;
                                row_sums.push_back(float(got));
                            }
                        }
                    }
                }
                sum_error = std::max(sum_error, this_sum_error);
                if (first.empty()) first = actual;
                else if (first != actual) throw std::runtime_error("Output changed between submissions");
            }
            if (name == "deterministic" && row_sums != std::vector<float>{6, 15})
                throw std::runtime_error("Hand-checked deterministic result mismatch");

            cases.push_back({{"name",name},{"shape_hw",{height,width}},{"padded_hw",{hp,wp}},
                {"tile_grid_hw",{ht,wt}},{"logical_output_shape",{height,1}},{"device_output_shape",{ht,32,32}},{"device_output_layout","transposed final prefix tile"},
                {"row_sums",row_sums},{"passed",true},{"repetitions",repeats},
                {"input_fingerprint",af::input_fingerprint(packed)},{"output_fingerprint",af::input_fingerprint(first)},
                {"layout_matches_official_tilizer",true},{"output_sentinel_overwritten",true},
                {"output_padded_lanes_zero",true},{"entire_final_prefix_tile_verified",true},{"all_submission_outputs_equal",true},
                {"max_abs_error",max_error},{"mean_abs_error",sum_error/height},
                {"atol",exact ? 0.0 : 1e-4},{"rtol",exact ? 0.0 : 1e-5},
                {"input_buffer",af::buffer_observation(xd)},{"output_buffer",af::buffer_observation(yd)},
                {"input_tile_reads_model",ht*wt},{"scaler_tile_reads_model",0},
                {"cumsum_tile_calls_model",ht*wt},{"transpose_tile_calls_model",ht*wt},{"destination_tile_slots_max",1},
                {"intermediate_pack_calls_model",0},{"output_tile_writes_model",ht},
                {"dram_read_payload_bytes_model",uint64_t(ht)*wt*2048},
                {"dram_write_payload_bytes_model",ht*4096},{"cb_capacity_bytes_model",depth*6144},
                {"core_count",1},{"model_policy","source-derived, not hardware counters"}});
            std::cout << "PASS row_sum " << name << ' ' << height << ',' << width
                      << " max_abs=" << max_error << " repeats=" << repeats << '\n';
        };

        run("deterministic",2,3);
        run("zero",33,35);
        for (auto width : {1U,31U,32U,33U,257U,1025U,4097U}) run("ones",2,width);
        for (auto height : {1U,31U,32U,33U,65U}) run("signed_random",height,65);
        run("signed_random",65,129);
        run("signed_random",17,1025);
        run("large_exact",2,1025);
        run("cancellation",3,3);
        run("cancellation",3,32);
        run("cancellation",3,1025);
        run("fractional",1,33);
        run("fractional",33,65);
        run("fractional",17,257);
        run("fractional",3,1025);
        run("fractional",2,4097);
        if (cases.empty()) throw std::runtime_error("No fixture matched selector");
        session.close();
        af::save(argv[1],{{"operator","row_sum"},{"backend","TTSim Wormhole"},
            {"implementation","sfpu_scan_row_sum"},{"input_dtype","BF16"},{"accumulation_dtype","FP32"},{"output_dtype","FP32"},
            {"reference","FP64 sum of quantized BF16 input"},{"seed",2026},
            {"cb_depth",depth},{"repeats",repeats},{"cases",cases},{"passed",true}});
        std::cout << "REDUCTION_ALL_PASS cases=" << cases.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
