#include "../vector_add/common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <string>

static size_t offset(uint32_t row, uint32_t col, uint32_t width) {
    const uint32_t r = row % 32, c = col % 32;
    return (row / 32 * (width / 32) + col / 32) * 1024 +
           (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16;
}

static std::vector<float> pack(const std::vector<float>& values, uint32_t h, uint32_t w) {
    std::vector<float> result(values.size());
    for (uint32_t row = 0; row < h; ++row)
        for (uint32_t col = 0; col < w; ++col)
            result[offset(row, col, w)] = values[row * w + col];
    return result;
}

static float widen(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6 && argc != 7)
            throw std::runtime_error("Usage: rmsnorm fixture-list.txt report.json depth repeats [unfused|fused] [cores:1|2|4]");
        const std::string variant = argc >= 6 ? argv[5] : "unfused";
        if (variant != "unfused" && variant != "fused")
            throw std::runtime_error("Expected unfused or fused");
        const bool fused = variant == "fused";
        const uint32_t requested_cores = argc == 7 ? std::stoul(argv[6]) : 1;
        if ((requested_cores != 1 && requested_cores != 2 && requested_cores != 4) ||
            (!fused && requested_cores != 1))
            throw std::runtime_error("Only fused supports 1/2/4 cores; unfused requires one core");
        const uint32_t depth = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
        if ((depth != 1 && depth != 2) || repeats < 1 || repeats > 3)
            throw std::runtime_error("Invalid depth/repeats");
        std::ifstream list(argv[1]);
        if (!list) throw std::runtime_error("Could not open fixture list");
        af::DeviceSession session;
        auto& queue = session.device->mesh_command_queue();
        nlohmann::json cases = nlohmann::json::array();
        std::string path;

        while (std::getline(list, path)) {
            if (path.empty()) continue;
            std::ifstream file(path, std::ios::binary);
            char magic[8]; uint32_t dims[4]{}; float epsilon = 0;
            file.read(magic, 8);
            file.read(reinterpret_cast<char*>(dims), 16);
            file.read(reinterpret_cast<char*>(&epsilon), 4);
            const uint32_t h = dims[0], w = dims[1], hp = dims[2], wp = dims[3];
            if (!file || std::string(magic, 8) != "AFRM0001" || !h || !w || h > 2048 || w > 8193 ||
                hp != (h + 31) / 32 * 32 || wp != (w + 31) / 32 * 32 || !(epsilon > 0))
                throw std::runtime_error("Invalid RMSNorm fixture");
            if (fused && w > 4097)
                throw std::runtime_error("Fused RMSNorm supports W <= 4097 to bound retained L1 storage");
            std::vector<uint16_t> bits(size_t(hp) * wp + wp);
            file.read(reinterpret_cast<char*>(bits.data()), bits.size() * 2);
            if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Fixture length mismatch");
            std::vector<float> x(hp * wp), gamma(32 * wp);
            for (size_t i = 0; i < x.size(); ++i) x[i] = widen(bits[i]);
            for (uint32_t row = 0; row < 32; ++row)
                for (uint32_t col = 0; col < wp; ++col) gamma[row * wp + col] = widen(bits[x.size() + col]);
            const uint32_t ht = hp / 32, wt = wp / 32, tiles = ht * wt;
            auto xd = af::buffer(session, tiles * 4096, 4096, af::BufferType::DRAM);
            auto gd = af::buffer(session, wt * 4096, 4096, af::BufferType::DRAM);
            auto square = fused ? decltype(xd){} : af::buffer(session, tiles * 4096, 4096, af::BufferType::DRAM);
            auto sums = fused ? decltype(xd){} : af::buffer(session, ht * 4096, 4096, af::BufferType::DRAM);
            auto yd = af::buffer(session, tiles * 4096, 4096, af::BufferType::DRAM);
            const auto packed_x = pack(x, hp, wp), packed_g = pack(gamma, 32, wp);
            af::dist::EnqueueWriteMeshBuffer(queue, xd, packed_x, false);
            af::dist::EnqueueWriteMeshBuffer(queue, gd, packed_g, false);

            // 각 코어는 연속한 32행 묶음을 맡습니다. 빈 shard에는 커널을 만들지 않습니다.
            const uint32_t active_cores = fused ? std::min(requested_cores, ht) : 1;
            const auto grid = session.device->compute_with_storage_grid_size();
            if (active_cores > grid.x * grid.y) throw std::runtime_error("Insufficient worker cores");
            nlohmann::json shards = nlohmann::json::array();
            std::vector<uint32_t> coverage(ht, 0);
            uint32_t group_start = 0;
            for (uint32_t index = 0; index < active_cores; ++index) {
                const uint32_t groups = ht / active_cores + (index < ht % active_cores ? 1 : 0);
                const uint32_t valid_rows = std::min(h - group_start * 32, groups * 32);
                shards.push_back({{"core",{index % grid.x,index / grid.x}},
                    {"row_group_start",group_start},{"row_group_count",groups},
                    {"tile_start",group_start*wt},{"tile_count",groups*wt},{"valid_rows",valid_rows}});
                for (uint32_t group = group_start; group < group_start + groups; ++group) ++coverage[group];
                group_start += groups;
            }
            if (group_start != ht || std::any_of(coverage.begin(),coverage.end(),[](auto n){return n != 1;}))
                throw std::runtime_error("Row-group assignment has a gap or overlap");

            // 세 stage 사이의 데이터는 장치 DRAM에 남습니다. Host는 중간 계산을 하지 않습니다.
            auto make_stage = [&](uint32_t stage) {
                auto program = af::CreateProgram();
                const af::CoreCoord core{0, 0};
                const std::vector<uint32_t> cbs = stage == 2 ? std::vector<uint32_t>{0,1,2,3,4,5,16} : std::vector<uint32_t>{0,16};
                for (auto cb : cbs) {
                    const uint32_t slots = cb == 3 ? 1 : depth;
                    af::CreateCircularBuffer(program, core,
                        af::CircularBufferConfig(slots * 4096, {{cb, af::DataFormat::Float32}}).set_page_size(cb, 4096));
                }
                af::DataMovementConfig reader{}, writer{};
                reader.processor = af::DataMovementProcessor::RISCV_0;
                reader.noc = af::NOC::RISCV_0_default;
                writer.processor = af::DataMovementProcessor::RISCV_1;
                writer.noc = af::NOC::RISCV_1_default;
                auto input = stage == 1 ? square : xd;
                auto output = stage == 0 ? square : stage == 1 ? sums : yd;
                af::TensorAccessorArgs(*input).append_to(reader.compile_args);
                if (stage == 2) {
                    af::TensorAccessorArgs(*gd).append_to(reader.compile_args);
                    af::TensorAccessorArgs(*sums).append_to(reader.compile_args);
                }
                af::TensorAccessorArgs(*output).append_to(writer.compile_args);
                af::ComputeConfig math{};
                math.math_fidelity = af::MathFidelity::HiFi4;
                math.fp32_dest_acc_en = true;
                math.math_approx_mode = false;

                const std::string dir = ACCELFLOW_KERNEL_DIR;
                auto rk = af::CreateKernel(program, dir + (stage == 2 ? "/reader_normalize.cpp" : stage == 1 ? "/reader_sum.cpp" : "/reader.cpp"), core, reader);
                auto ck = af::CreateKernel(program, dir + (stage == 0 ? "/square.cpp" : stage == 1 ? "/sum.cpp" : "/normalize.cpp"), core, math);
                auto wk = af::CreateKernel(program, dir + "/writer.cpp", core, writer);
                if (stage == 2) {
                    af::SetRuntimeArgs(program, rk, core, {af::address32(xd->address()), af::address32(gd->address()), af::address32(sums->address()), ht, wt});
                    af::SetRuntimeArgs(program, ck, core, {tiles, std::bit_cast<uint32_t>(1.0f / w), std::bit_cast<uint32_t>(epsilon)});
                } else {
                    if (stage == 1) af::SetRuntimeArgs(program, rk, core, {af::address32(input->address()), ht, wt});
                    else af::SetRuntimeArgs(program, rk, core, {af::address32(input->address()), tiles});
                    if (stage == 0) af::SetRuntimeArgs(program, ck, core, {tiles});
                    else af::SetRuntimeArgs(program, ck, core, {ht, wt});
                }
                af::SetRuntimeArgs(program, wk, core, {af::address32(output->address()), stage == 1 ? ht : tiles});
                af::dist::MeshWorkload workload;
                workload.add_program(af::dist::MeshCoordinateRange(session.device->shape()), std::move(program));
                return workload;
            };
            auto make_fused = [&] {
                auto program = af::CreateProgram();
                for (const auto& shard : shards) {
                const af::CoreCoord core{shard["core"][0].get<uint32_t>(),shard["core"][1].get<uint32_t>()};
                const uint32_t row_start = shard["row_group_start"].get<uint32_t>();
                const uint32_t row_count = shard["row_group_count"].get<uint32_t>();
                // 같은 row group의 입력과 제곱 타일을 각각 L1에 보관합니다.
                for (uint32_t cb : {0U,1U,2U,4U,5U,16U,24U,25U,26U}) {
                    const uint32_t slots = cb == 24 || cb == 25 ? wt : cb == 26 ? 1 : depth;
                    af::CreateCircularBuffer(program, core,
                        af::CircularBufferConfig(slots * 4096, {{cb, af::DataFormat::Float32}})
                            .set_page_size(cb, 4096));
                }
                af::DataMovementConfig reader{}, writer{};
                reader.processor = af::DataMovementProcessor::RISCV_0;
                reader.noc = af::NOC::RISCV_0_default;
                writer.processor = af::DataMovementProcessor::RISCV_1;
                writer.noc = af::NOC::RISCV_1_default;
                af::TensorAccessorArgs(*xd).append_to(reader.compile_args);
                af::TensorAccessorArgs(*gd).append_to(reader.compile_args);
                af::TensorAccessorArgs(*yd).append_to(writer.compile_args);
                af::ComputeConfig math{};
                math.math_fidelity = af::MathFidelity::HiFi4;
                math.fp32_dest_acc_en = true;
                math.math_approx_mode = false;
                const std::string dir = ACCELFLOW_KERNEL_DIR;
                auto rk = af::CreateKernel(program, dir + "/fused_reader.cpp", core, reader);
                auto ck = af::CreateKernel(program, dir + "/fused_compute.cpp", core, math);
                auto wk = af::CreateKernel(program, dir + "/fused_writer.cpp", core, writer);
                af::SetRuntimeArgs(program, rk, core,
                    {af::address32(xd->address()), af::address32(gd->address()), row_count, wt, row_start});
                af::SetRuntimeArgs(program, ck, core,
                    {row_count, wt, std::bit_cast<uint32_t>(1.0f / w), std::bit_cast<uint32_t>(epsilon)});
                af::SetRuntimeArgs(program, wk, core, {af::address32(yd->address()), row_count*wt, row_start*wt});
                }
                af::dist::MeshWorkload workload;
                workload.add_program(af::dist::MeshCoordinateRange(session.device->shape()), std::move(program));
                return workload;
            };

            const uint32_t cb_bytes = (fused ? 2*wt+6*depth+1 : 6*depth+1) * 4096;
            if (fused && cb_bytes > 1200U * 1024)
                throw std::runtime_error("Fused CB payload exceeds the conservative 1200 KiB budget");
            std::vector<af::dist::MeshWorkload> workloads;
            if (fused) {
                workloads.push_back(make_fused());
            } else {
                for (uint32_t stage = 0; stage < 3; ++stage)
                    workloads.push_back(make_stage(stage));
            }
            std::vector<double> sum_reference(hp, 0), reference(x.size(), 0);
            for (uint32_t row = 0; row < h; ++row) {
                for (uint32_t col = 0; col < w; ++col) {
                    const double v = x[row * wp + col];
                    sum_reference[row] += v * v;
                }
                const double scale = 1 / std::sqrt(sum_reference[row] / w + double(epsilon));
                for (uint32_t col = 0; col < w; ++col)
                    reference[row * wp + col] = double(x[row * wp + col]) * scale * gamma[col];
            }
            const std::vector<float> sentinel(x.size(), std::numeric_limits<float>::quiet_NaN());
            const std::vector<float> stat_sentinel(ht * 1024, std::numeric_limits<float>::quiet_NaN());
            double max_error = 0, max_sum_error = 0;
            std::vector<float> first;
            for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
                if (!fused) {
                    af::dist::EnqueueWriteMeshBuffer(queue, square, sentinel, false);
                    af::dist::EnqueueWriteMeshBuffer(queue, sums, stat_sentinel, false);
                }
                af::dist::EnqueueWriteMeshBuffer(queue, yd, sentinel, false);
                for (auto& work : workloads) {
                    af::dist::EnqueueMeshWorkload(queue, work, false);
                    af::dist::Finish(queue);
                }
                std::vector<float> actual, squared, statistics;
                af::dist::EnqueueReadMeshBuffer(queue, actual, yd, true);
                if (!fused) {
                    af::dist::EnqueueReadMeshBuffer(queue, squared, square, true);
                    af::dist::EnqueueReadMeshBuffer(queue, statistics, sums, true);
                }
                for (uint32_t row = 0; row < hp; ++row) {
                    for (uint32_t col = 0; col < wp; ++col) {
                        const size_t i = offset(row, col, wp), logical = row * wp + col;
                        const double expected = reference[logical], got = actual[i];
                        const double error = std::abs(got - expected);
                        if (!std::isfinite(got) || error > 1e-4 + 1e-4 * std::abs(expected) ||
                            ((row >= h || col >= w) && got != 0))
                            throw std::runtime_error(path + ": normalize mismatch " + std::to_string(row) + "," + std::to_string(col) + " expected=" + std::to_string(expected) + " got=" + std::to_string(got) + " sum_sq=" + std::to_string(fused ? 0.0f : statistics[(row / 32) * 1024 + offset(31, row % 32, 32)]) + " expected_sum_sq=" + std::to_string(sum_reference[row]));
                        if (!fused && squared[i] != x[logical] * x[logical]) throw std::runtime_error("Square stage mismatch");
                        max_error = std::max(max_error, error);
                    }
                    if (!fused) {
                    const float got_sum = statistics[(row / 32) * 1024 + offset(31, row % 32, 32)];
                    const double err = std::abs(double(got_sum) - sum_reference[row]);
                    if (!std::isfinite(got_sum) || err > 1e-4 + 1e-4 * sum_reference[row])
                        throw std::runtime_error("Sum-square stage mismatch");
                    max_sum_error = std::max(max_sum_error, err);
                    }
                }
                if (first.empty()) first = actual;
                else if (first != actual) throw std::runtime_error("Repeated outputs differ");
            }
            std::vector<float> logical_output;
            for (uint32_t row=0;row<h;++row) for(uint32_t col=0;col<w;++col) logical_output.push_back(first[offset(row,col,wp)]);
            cases.push_back({{"logical_output",logical_output},{"key",std::filesystem::path(path).stem().string()},{"shape_hw",{h,w}},
                {"epsilon",epsilon},{"passed",true},{"repeats",repeats},{"max_abs_error",max_error},
                {"max_sum_square_error",fused ? nlohmann::json(nullptr) : nlohmann::json(max_sum_error)},
                {"square_exact",fused ? nlohmann::json(nullptr) : nlohmann::json(true)},
                {"intermediate_dram_checked",!fused},{"padding_zero",true},
                {"input_fingerprint",af::input_fingerprint(packed_x)},{"gamma_fingerprint",af::input_fingerprint(packed_g)},
                {"output_fingerprint",af::input_fingerprint(first)},
                {"cb_peak_payload_bytes_model",cb_bytes},{"device_l1_bytes_per_core",session.device->l1_size_per_core()},
                {"intermediate_dram_bytes",fused ? 0U : (tiles+ht)*4096},
                {"dram_payload_bytes_model",uint64_t(fused ? 3*tiles : 7*tiles+2*ht)*4096},
                {"row_cache_bytes_model",fused ? 2*wt*4096 : 0U},{"variant",variant},
                {"requested_cores",requested_cores},{"active_cores",active_cores},
                {"idle_requested_cores",requested_cores-active_cores},{"shards",shards},
                {"cb_total_payload_bytes_model",uint64_t(cb_bytes)*active_cores},
                {"assignment_has_no_gaps_or_overlaps",true}});
            std::cout << "PASS RMSNorm " << path << " max_abs=" << max_error << std::endl;
        }
        if (cases.empty()) throw std::runtime_error("No fixtures");
        session.close();
        af::save(argv[2],{{"passed",true},{"backend","TTSim Wormhole"},{"variant",variant},{"stages",fused ? 1 : 3},
            {"core_count",requested_cores},{"core_count_policy","requested; per-case active_cores skips empty shards"},{"cb_depth",depth},{"atol",1e-4},{"rtol",1e-4},{"cases",cases}});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
