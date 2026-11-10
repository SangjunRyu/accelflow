#include "../vector_add/common.hpp"
#include "benchmark.hpp"
#include "multicore.hpp"
#include "mcast.hpp"
#include "mcast2d.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <filesystem>
#include <fstream>

// 행렬 차원을 32의 배수로 올려, 가장자리도 완전한 타일로 처리합니다.
static uint32_t padded(uint32_t size) {
    return ((size + 31) / 32) * 32;
}

// 행/열 좌표 → 타일 번호 → 16×16 face 번호 → face 내부 위치로 변환합니다.
static size_t offset(uint32_t row, uint32_t col, uint32_t columns) {
    const uint32_t r = row % 32, c = col % 32;
    return (row / 32 * (columns / 32) + col / 32) * 1024 + (r / 16 * 2 + c / 16) * 256 +
           r % 16 * 16 + c % 16;
}

// CPU row-major 입력을 장치의 face 배치로 변환하고 공식 tilizer와 대조합니다.
static std::vector<bfloat16> pack(const std::vector<bfloat16>& row_major,
                                  uint32_t rows,
                                  uint32_t columns) {
    std::vector<bfloat16> result(row_major.size());
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < columns; ++c)
            result[offset(r, c, columns)] = row_major[r * columns + c];
    const auto upstream = tilize_nfaces(row_major, rows, columns);
    if (result != upstream)
        throw std::runtime_error("Own tile layout differs from official tilize_nfaces");
    return result;
}

// Host 흐름: 입력 준비 → DRAM 전송 → 커널/CB 구성 → 실행 → CPU 비교 → 보고서 저장.
int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5 && argc != 8 && argc != 10)
            throw std::runtime_error("Usage: matmul report.json baseline|reuse_a|blocked|streamed|multicore|mcast|mcast2d|mcast2d_batch1|mcast2d_batch4|mcast2d_batch8 1|2 [K-chunk: 1|2, or multicore cores: 1|2|4] [--benchmark samples name:M,K,N [--export-inputs fixture.bin]]");

        // 실험 설정: baseline / A 행 재사용 / 2×2 blocked, CB 깊이 1/2.
        const std::string variant = argv[2];
        if (variant != "baseline" && variant != "reuse_a" &&
            variant != "blocked" && variant != "streamed" && variant != "multicore" && variant != "mcast" && variant != "mcast2d" && variant != "mcast2d_batch1" && variant != "mcast2d_batch4" && variant != "mcast2d_batch8")
            throw std::runtime_error("Invalid variant");
        const bool mcast = variant == "mcast";
        // batch1은 새 루프 자체의 영향을 비교하는 대조군입니다.
        const bool batched = variant.rfind("mcast2d_batch", 0) == 0;
        const uint32_t batch = batched ? std::stoul(variant.substr(13)) : 1;
        const bool mcast2d = variant == "mcast2d" || batched;
        const bool multicore = variant == "multicore" || mcast || mcast2d;
        const uint32_t requested_cores = multicore && argc >= 5 ? std::stoul(argv[4]) : 4;
        if (multicore && requested_cores != 1 && requested_cores != 2 && requested_cores != 4)
            throw std::runtime_error("Multicore requests must be 1, 2, or 4");
        const bool reuse = variant == "reuse_a";
        const bool streamed = variant == "streamed";
        const bool blocked = variant == "blocked" || streamed;
        const uint32_t chunk = argc >= 5 && !multicore ? std::stoul(argv[4]) : 2;
        const bool benchmarking = argc == 8 || argc == 10;
        const bool export_inputs = argc == 10;
        if (export_inputs && std::string(argv[8]) != "--export-inputs")
            throw std::runtime_error("Expected --export-inputs fixture.bin after benchmark selector");
        const uint32_t iterations = benchmarking ? std::stoul(argv[6]) : 0;
        if (benchmarking && (std::string(argv[5]) != "--benchmark" ||
                            iterations < 5 || iterations > 25))
            throw std::runtime_error("Benchmark requires 5..25 samples");
        if ((argc == 5 && !streamed && !multicore) || (chunk != 1 && chunk != 2))
            throw std::runtime_error("K chunk 1/2 applies only to streamed mode");
        const uint32_t depth = std::stoul(argv[3]);
        if (depth != 1 && depth != 2)
            throw std::runtime_error("CB depth must be 1 or 2");
        af::DeviceSession session;
        std::mt19937 rng(2026);
        nlohmann::json cases = nlohmann::json::array();

        // 하나의 입력 사례를 실행합니다. 모든 설정에서 같은 seed와 사례 순서를 사용합니다.
        auto run = [&](std::string name, uint32_t m, uint32_t k, uint32_t n) {
            // benchmark는 동일 shape 하나만 실행하여 다른 사례의 준비 비용을 제외합니다.
            if (benchmarking && std::string(argv[7]) !=
                name + ":" + std::to_string(m) + "," + std::to_string(k) + "," + std::to_string(n))
                return;

            // m/k/n은 실제 크기, mp/kp/np는 padding 크기, mt/kt/nt는 타일 개수입니다.
            const uint32_t mp = padded(m), kp = padded(k), np = padded(n), mt = mp / 32,
                           kt = kp / 32, nt = np / 32;
            // blocked는 K 전체를 보관하므로 첫 구현의 검증 범위를 Kt <= 4로 제한합니다.
            if (blocked && !streamed && kt > 4)
                throw std::runtime_error("Blocked MatMul supports Kt <= 4");

            std::vector<bfloat16> a(mp * kp, bfloat16(0.0f)), b(kp * np, bfloat16(0.0f));
            std::uniform_int_distribution<int> integer(-1, 1);
            std::uniform_real_distribution<float> fraction(-0.5f, 0.5f);

            // 유효 영역만 채우고 padding은 0으로 유지합니다. CPU 기준값도 BF16 입력을 사용합니다.
            for (uint32_t r = 0; r < m; ++r)
                for (uint32_t col = 0; col < k; ++col) {
                    float value = name == "fractional" ? fraction(rng) : float(integer(rng));
                    if (name == "zero")
                        value = 0;
                    if (name == "identity")
                        value = r == col ? 1.0f : 0.0f;
                    if (name == "large_exact")
                        value = 8;
                    if (name == "deterministic")
                        value = float(r * k + col + 1);
                    a[r * kp + col] = bfloat16(value);
                }
            for (uint32_t r = 0; r < k; ++r)
                for (uint32_t col = 0; col < n; ++col) {
                    float value = name == "fractional" ? fraction(rng) : float(integer(rng));
                    if (name == "large_exact")
                        value = 8;
                    if (name == "deterministic")
                        value = float(r * n + col + 7);
                    b[r * np + col] = bfloat16(value);
                }

            // 1. 입력을 장치 타일 배치로 변환하고 DRAM에 올립니다. 한 BF16 타일은 2048 bytes입니다.
            const auto pa = pack(a, mp, kp), pb = pack(b, kp, np);
            // 같은 BF16 입력을 CUDA에도 전달합니다. MSVC/libstdc++의 난수 분포 차이를 피합니다.
            // 파일은 little-endian: magic 8 bytes, M/K/N 각 uint32, padded row-major A/B BF16.
            if (export_inputs) {
                static_assert(sizeof(bfloat16) == 2);
                const std::filesystem::path path(argv[9]);
                if (!path.parent_path().empty())
                    std::filesystem::create_directories(path.parent_path());
                std::ofstream fixture(path, std::ios::binary);
                fixture.write("AFMM001", 8);
                fixture.write(reinterpret_cast<const char*>(&m), 4);
                fixture.write(reinterpret_cast<const char*>(&k), 4);
                fixture.write(reinterpret_cast<const char*>(&n), 4);
                fixture.write(reinterpret_cast<const char*>(a.data()), a.size() * 2);
                fixture.write(reinterpret_cast<const char*>(b.data()), b.size() * 2);
                fixture.close();
                if (!fixture)
                    throw std::runtime_error("Cannot export MatMul input fixture");
            }

            auto ad = af::buffer(session, pa.size() * 2, 2048, af::BufferType::DRAM);
            auto bd = af::buffer(session, pb.size() * 2, 2048, af::BufferType::DRAM);
            auto cd = af::buffer(session, mp * np * 2, 2048, af::BufferType::DRAM);
            auto& queue = session.device->mesh_command_queue();
            af::dist::EnqueueWriteMeshBuffer(queue, ad, pa, false);
            af::dist::EnqueueWriteMeshBuffer(queue, bd, pb, false);

            // 출력을 NaN으로 초기화해 미기록 영역이 우연히 정답처럼 보이는 것을 막습니다.
            const std::vector<bfloat16> sentinel(mp * np,
                                                 bfloat16(std::numeric_limits<float>::quiet_NaN()));
            af::dist::EnqueueWriteMeshBuffer(queue, cd, sentinel, false);

            // 2. 프로그램과 코어별 L1 circular buffer를 구성합니다: CB0=A, CB1=B, CB16=C.
            auto program = af::CreateProgram();
            // reuse_a는 A의 Kt개 타일을 보관하고, blocked는 A/B 두 행·열을 보관합니다.
            // streamed는 2×chunk, blocked는 2×Kt 입력 슬롯을 depth개 보관합니다.
            const uint32_t input_block_k = streamed ? chunk : kt;
            const uint32_t a_capacity =
                blocked ? 2 * input_block_k * depth : (reuse ? kt : batch * depth);
            const uint32_t b_capacity = blocked ? 2 * input_block_k * depth : batch * depth;

            nlohmann::json distribution = nullptr;
            if (mcast2d) {
                distribution = af::mcast2d::configure(
                    session, program, ad, bd, cd, mt, kt, nt, depth, requested_cores, batch, batched);
            } else if (mcast) {
                distribution = af::mcast::configure(
                    session, program, ad, bd, cd, mt, kt, nt, depth, requested_cores);
            } else if (multicore) {
                distribution = af::multicore::configure(
                    session, program, ad, bd, cd, mt, kt, nt, depth, requested_cores);
            } else {
                const af::CoreCoord core{0, 0};

                for (auto index : {af::CBIndex::c_0, af::CBIndex::c_1, af::CBIndex::c_16}) {
                    const uint32_t capacity =
                        index == af::CBIndex::c_0 ? a_capacity :
                        (index == af::CBIndex::c_1 ? b_capacity : depth);
                    af::CreateCircularBuffer(
                        program,
                        core,
                        af::CircularBufferConfig(capacity * 2048, {{index, af::DataFormat::Float16_b}})
                            .set_page_size(index, 2048));
                }

                // 3. Reader는 입력 이동, compute는 Tensix 행렬곱, writer는 결과 이동을 담당합니다.
                af::DataMovementConfig reader_config{}, writer_config{};
                reader_config.processor = af::DataMovementProcessor::RISCV_1;
                reader_config.noc = af::NOC::RISCV_1_default;
                af::TensorAccessorArgs(*ad).append_to(reader_config.compile_args);
                af::TensorAccessorArgs(*bd).append_to(reader_config.compile_args);
                writer_config.processor = af::DataMovementProcessor::RISCV_0;
                writer_config.noc = af::NOC::RISCV_0_default;
                af::TensorAccessorArgs(*cd).append_to(writer_config.compile_args);
                af::ComputeConfig compute_config{};
                compute_config.math_fidelity = af::MathFidelity::HiFi4;
                // 큰 K의 누적 오차를 줄입니다. 모든 비교 모드에 동일하게 적용합니다.
                compute_config.fp32_dest_acc_en = true;
                compute_config.compile_args = {reuse ? 1U : 0U};
                const std::string kernels = ACCELFLOW_KERNEL_DIR;
                const std::string suffix = streamed ? "_streamed.cpp" :
                    (blocked ? "_blocked.cpp" : ".cpp");
                auto reader =
                    af::CreateKernel(program, kernels + "/reader" + suffix, core, reader_config);
                auto writer =
                    af::CreateKernel(program, kernels + (blocked ? "/writer_blocked.cpp" : "/writer.cpp"), core, writer_config);
                auto compute =
                    af::CreateKernel(program, kernels + "/compute" + suffix, core, compute_config);

                // 주소와 shape를 전달하는 순서는 각 커널의 get_arg_val 인덱스와 같아야 합니다.
                af::SetRuntimeArgs(program,
                                   reader,
                                   core,
                                   {af::address32(ad->address()),
                                    af::address32(bd->address()),
                                    mt,
                                    kt,
                                    nt,
                                    reuse ? 1U : 0U,
                                    chunk});
                if (blocked) {
                    af::SetRuntimeArgs(program, writer, core, {af::address32(cd->address()), mt, nt});
                } else {
                    af::SetRuntimeArgs(program, writer, core, {af::address32(cd->address()), mt * nt});
                }
                af::SetRuntimeArgs(program, compute, core, {mt, kt, nt, chunk});

            }

            // 4. 프로그램 완료를 기다린 뒤 결과를 host로 읽습니다.
            std::vector<bfloat16> actual;
            nlohmann::json timing = nullptr;
            if (benchmarking) {
                timing = af::benchmark::repeat(
                    session, std::move(program), cd, sentinel, iterations, actual);
            } else {
                af::execute(session, std::move(program));
                af::dist::EnqueueReadMeshBuffer(queue, actual, cd, true);
            }
            if (actual.size() != mp * np)
                throw std::runtime_error("Output size mismatch");

            // 5. 양자화된 입력의 곱을 FP64로 누적해 비교합니다. 출력 padding까지 전부 검사합니다.
            double max_error = 0, sum_error = 0, max_relative = 0;

            // 정수 fixture는 exact 비교, 소수 fixture는 사전에 정한 절대/상대 허용 오차를
            // 적용합니다.
            const bool exact = name != "fractional";
            for (uint32_t r = 0; r < mp; ++r)
                for (uint32_t col = 0; col < np; ++col) {
                    double reference = 0;
                    if (r < m && col < n)
                        for (uint32_t inner = 0; inner < k; ++inner)
                            reference +=
                                double(float(a[r * kp + inner])) * float(b[inner * np + col]);
                    const double got = float(actual[offset(r, col, np)]),
                                 error = std::abs(got - reference);
                    const bool tail = r >= m || col >= n;
                    const double tolerance = exact ? 0.0 : 0.03 + 0.03 * std::abs(reference);
                    if (!std::isfinite(got) || error > tolerance || (tail && got != 0))
                        throw std::runtime_error(name + ": mismatch at (" + std::to_string(r) +
                                                 "," + std::to_string(col) +
                                                 ") expected=" + std::to_string(reference) +
                                                 " actual=" + std::to_string(got));
                    if (!tail) {
                        max_error = std::max(max_error, error);
                        sum_error += error;
                        max_relative =
                            std::max(max_relative, error / std::max(1e-6, std::abs(reference)));
                    }
                }

            // 작은 2×3 · 3×2 사례는 손으로 계산한 기대값과도 별도로 비교합니다.
            if (name == "deterministic") {
                const std::vector<float> expected{58, 64, 139, 154};
                for (uint32_t i = 0; i < 4; ++i)
                    if (float(actual[offset(i / 2, i % 2, np)]) != expected[i])
                        throw std::runtime_error("Deterministic reference mismatch");
            }

            // 6. 루프 구조에서 전송량/연산량 모델을 계산합니다. 실물 성능 counter가 아닙니다.
            const uint64_t products = uint64_t(mt) * kt * nt;
            const uint64_t m_blocks = (mt + 1) / 2, n_blocks = (nt + 1) / 2;
            const uint64_t a_loads =
                (mcast || mcast2d) ? uint64_t(distribution["a_dram_reads_model"]) :
                blocked ? uint64_t(mt) * kt * n_blocks :
                (reuse ? uint64_t(mt) * kt : products);
            const uint64_t b_loads = mcast2d ? uint64_t(distribution["b_dram_reads_model"]) :
                (blocked ? uint64_t(nt) * kt * m_blocks : products);
            cases.push_back({{"name", name},
                             {"timing", timing},
                             {"distribution", distribution},
                             {"A_multicast_enabled", mcast || mcast2d},
                             {"B_multicast_enabled", mcast2d},
                             {"core_count", multicore ? uint32_t(distribution["active_cores"]) : 1},
                             {"cb_capacity_bytes_total_model", (a_capacity + b_capacity + depth) * 2048ULL *
                                 (multicore ? uint32_t(distribution["active_cores"]) : 1)},
                             {"shape_mkn", {m, k, n}},
                             {"padded_mkn", {mp, kp, np}},
                             {"tile_grid_mkn", {mt, kt, nt}},
                             {"a_shape", {m, k}},
                             {"b_shape", {k, n}},
                             {"c_shape", {m, n}},
                             {"fp32_destination_accumulation", true},
                             {"passed", true},
                             {"layout_matches_official_tilizer", true},
                             {"output_sentinel_overwritten", true},
                             {"padding_verified", true},
                             {"max_abs_error", max_error},
                             {"mean_abs_error", sum_error / (m * n)},
                             {"max_relative_error", max_relative},
                             {"atol", exact ? 0.0 : 0.03},
                             {"rtol", exact ? 0.0 : 0.03},
                             {"input_fingerprint_a", af::input_fingerprint(pa)},
                             {"input_fingerprint_b", af::input_fingerprint(pb)},
                             {"output_fingerprint", af::input_fingerprint(actual)},
                             {"a_buffer", af::buffer_observation(ad)},
                             {"b_buffer", af::buffer_observation(bd)},
                             {"c_buffer", af::buffer_observation(cd)},
                             {"a_tile_reads_model", a_loads},
                             {"b_tile_reads_model", b_loads},
                             {"output_tile_writes_model", mt * nt},
                             {"dram_read_payload_model", (a_loads + b_loads) * 2048},
                             {"dram_write_payload_model", mt * nt * 2048},
                             {"matmul_tile_calls_model", products},
                             {"cb_capacity_tiles", {a_capacity, b_capacity, depth}},
                             {"cb_capacity_bytes_model", (a_capacity + b_capacity + depth) * 2048},
                             {"output_block_tiles", blocked ? 4 : 1},
                             {"destination_tile_slots_max", blocked ? 4 : 1},
                             {"k_chunk_tiles", streamed ? chunk : 0},
                             {"input_chunks_per_output_block", streamed ? (kt + chunk - 1) / chunk : 1},
                             {"partial_result_pack_calls_model", 0},
                             {"read_barriers_model", mcast2d ? uint64_t(distribution["read_barriers_model"]) : mcast ? a_loads + b_loads : blocked ? m_blocks * n_blocks *
                                 (streamed ? (kt + chunk - 1) / chunk : 1) :
                                 (reuse ? uint64_t(mt) + products : 2 * products)},
                             {"useful_flops_model", 2ULL * m * k * n},
                             {"padded_flops_model", 2ULL * mp * kp * np},
                             {"model_policy", "code-derived counts, not hardware counters"}});
            std::cout << "PASS " << variant << " depth=" << depth << " " << name << " MKN=" << m
                      << "," << k << "," << n << " max_abs=" << max_error << '\n';
        };

        // 검증 목록: 작은 행렬, zero/identity, 타일 경계, K축 누적, BF16 소수 오차.
        run("deterministic", 2, 3, 2);
        run("zero", 33, 35, 31);
        run("identity", 32, 32, 32);
        run("signed_random", 32, 32, 32);
        run("signed_random", 31, 33, 35);
        run("signed_random", 33, 31, 65);
        run("signed_random", 64, 64, 96);
        run("signed_random", 32, 128, 96);
        run("signed_random", 65, 97, 33);
        run("fractional", 32, 64, 64);
        run("fractional", 33, 35, 65);
        run("large_exact", 8, 16, 8);
        // 3×5 타일 경계와 반복 블록/CB wrap을 추가로 검증합니다.
        run("signed_random", 96, 64, 160);
        run("signed_random", 128, 128, 128);

        // K>128은 기존 blocked 범위 밖이므로 baseline/reuse_a/streamed로 비교합니다.
        if (!blocked || streamed) {
            run("signed_random", 65, 129, 97);
            run("signed_random", 65, 257, 97);
            run("signed_random", 64, 512, 64);
            run("fractional", 33, 161, 65);
            run("fractional", 33, 257, 65);
            run("zero", 33, 257, 65);
        }

        // 1×4 / 4×1 multicast 경로의 선택 실행용 fixture입니다. 기본 suite는 유지합니다.
        if (benchmarking) {
            run("signed_random", 32, 64, 128);
            run("signed_random", 128, 64, 32);
            run("signed_random", 320, 320, 320);
            run("signed_random", 640, 320, 640);
        }

        // 모든 사례를 통과한 경우에만 성공 보고서를 저장합니다.
        if (cases.empty())
            throw std::runtime_error("No supported benchmark fixture matched shape");
        session.close();
        af::save(argv[1],
                 {{"variant", variant},
                  {"benchmark_enabled", benchmarking},
                  {"requested_cores", multicore ? requested_cores : 1},
                  {"stream_cb_depth", depth},
                  {"multicast_batch_tiles", batch},
                  {"k_chunk_tiles", streamed ? chunk : 0},
                  {"dtype", "bf16"},
                  {"destination_accumulation", "fp32"},
                  {"seed", 2026},
                  {"reference", "FP64 accumulation of products of quantized BF16 inputs"},
                  {"cases", cases},
                  {"passed", true}});
        std::cout << "MATMUL_ALL_PASS cases=" << cases.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
