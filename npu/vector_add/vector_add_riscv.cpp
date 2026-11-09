#include "common.hpp"
#include <limits>
#include <random>

// uint32 기준 버전: RISC-V가 원소마다 읽기 → 덧셈 → 쓰기를 수행합니다.
int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Usage: vector_add_riscv report.json");
        af::DeviceSession session;
        std::mt19937 rng(2026);
        nlohmann::json cases = nlohmann::json::array();

        // 하나의 입력 사례를 장치에서 실행하고 host의 modulo 2^32 기준값과 비교합니다.
        auto run = [&](const std::string& name,
                       const std::vector<uint32_t>& a,
                       const std::vector<uint32_t>& b) {
            const uint32_t n = a.size();
            if (n == 0 || b.size() != n)
                throw std::runtime_error("Invalid input length");

            // DRAM은 전체 벡터, L1 scratch는 각 입력/출력 한 원소를 담습니다.
            auto ad = af::buffer(session, n * 4, 4, af::BufferType::DRAM);
            auto bd = af::buffer(session, n * 4, 4, af::BufferType::DRAM);
            auto cd = af::buffer(session, n * 4, 4, af::BufferType::DRAM);
            auto al = af::buffer(session, 4, 4, af::BufferType::L1);
            auto bl = af::buffer(session, 4, 4, af::BufferType::L1);
            auto cl = af::buffer(session, 4, 4, af::BufferType::L1);
            auto& cq = session.device->mesh_command_queue();
            af::dist::EnqueueWriteMeshBuffer(cq, ad, a, false);
            af::dist::EnqueueWriteMeshBuffer(cq, bd, b, false);

            // 단일 data-movement RISC-V 커널을 구성합니다. Tensix 타일 연산은 사용하지 않습니다.
            auto program = af::CreateProgram();
            const af::CoreCoord core{0, 0};
            af::DataMovementConfig kernel_config{};
            kernel_config.processor = af::DataMovementProcessor::RISCV_0;
            kernel_config.noc = af::NOC::RISCV_0_default;
            auto kernel =
                af::CreateKernel(program,
                                 std::string(ACCELFLOW_KERNEL_DIR) + "/vector_add_riscv.cpp",
                                 core,
                                 kernel_config);

            // 커널과 동일한 순서로 DRAM 주소, L1 주소, 원소 수를 전달합니다.
            af::SetRuntimeArgs(program,
                               kernel,
                               core,
                               {af::address32(ad->address()),
                                af::address32(bd->address()),
                                af::address32(cd->address()),
                                af::address32(al->address()),
                                af::address32(bl->address()),
                                af::address32(cl->address()),
                                n});

            // 장치 완료를 기다린 뒤 출력 전체를 읽어 exact 비교합니다.
            af::execute(session, std::move(program));
            std::vector<uint32_t> actual;
            af::dist::EnqueueReadMeshBuffer(cq, actual, cd, true);
            if (actual.size() != n)
                throw std::runtime_error("Output length mismatch");
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t expected = a[i] + b[i]; // Defined modulo 2^32.
                if (actual[i] != expected)
                    throw std::runtime_error(name + ": mismatch at " + std::to_string(i));
            }
            std::cout << "PASS riscv " << name << " n=" << n << '\n';

            // 보고서의 전송량/barrier 수는 원소별 루프에서 계산한 구조 모델입니다.
            cases.push_back({{"name", name},
                             {"length", n},
                             {"passed", true},
                             {"input_fingerprint_a", af::input_fingerprint(a)},
                             {"input_fingerprint_b", af::input_fingerprint(b)},
                             {"dram_buffer_observation", af::buffer_observation(ad)},
                             {"l1_scratch_observation", af::buffer_observation(al)},
                             {"dram_read_payload_bytes", 8ULL * n},
                             {"dram_write_payload_bytes", 4ULL * n},
                             {"logical_l1_scratch_bytes", 12},
                             {"kernel_count", 1},
                             {"core_count", 1},
                             {"read_barrier_calls", n},
                             {"store_visibility_checks", n},
                             {"write_barrier_calls", n}});
        };

        // 작은 입력, 0, unsigned overflow 및 경계 크기의 seeded random을 검증합니다.
        run("deterministic", {1, 2, 3, 4}, {5, 6, 7, 8});
        run("zero", std::vector<uint32_t>(33, 0), std::vector<uint32_t>(33, 0));
        run("unsigned_wrap", {UINT32_MAX, UINT32_MAX, 0, 123}, {1, UINT32_MAX, 0, 456});
        for (uint32_t n : {1U, 31U, 32U, 33U, 1024U, 1025U, 8193U}) {
            std::vector<uint32_t> a(n), b(n);
            for (uint32_t i = 0; i < n; ++i) {
                a[i] = rng();
                b[i] = rng();
            }
            run("seeded_random", a, b);
        }
        session.close();
        af::save(argv[1],
                 {{"variant", "riscv_scalar_loop"},
                  {"dtype", "uint32"},
                  {"seed", 2026},
                  {"comparison", "exact modulo 2^32"},
                  {"cases", cases},
                  {"passed", true}});
        std::cout << "RISCV_VECTOR_ADD_ALL_PASS cases=" << cases.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
