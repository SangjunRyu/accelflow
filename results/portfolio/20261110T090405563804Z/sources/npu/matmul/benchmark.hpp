#pragma once

#include "../vector_add/common.hpp"
#include <chrono>

namespace af::benchmark {

// steady_clock은 host 경과 시간을 측정합니다. 실제 NPU의 cycle counter가 아닙니다.
using Clock = std::chrono::steady_clock;

inline double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
nlohmann::json repeat(DeviceSession& session,
                      Program&& program,
                      std::shared_ptr<dist::MeshBuffer>& output,
                      const std::vector<T>& sentinel,
                      uint32_t iterations,
                      std::vector<T>& actual) {
    dist::MeshWorkload workload;
    workload.add_program(dist::MeshCoordinateRange(session.device->shape()), std::move(program));
    auto& queue = session.device->mesh_command_queue();
    std::vector<double> warmup_ms, samples_ms;
    std::vector<T> reference_output;

    // 같은 workload를 반복 제출합니다. host/device 수명과 버퍼 주소를 유지합니다.
    for (uint32_t i = 0; i < iterations + 2; ++i) {
        // 매번 NaN sentinel로 재설정해 이전 결과 재사용에 의한 가짜 성공을 막습니다.
        // 최초 입력 전송과 이 초기화는 측정 전에 반드시 완료합니다.
        dist::EnqueueWriteMeshBuffer(queue, output, sentinel, false);
        dist::Finish(queue);

        const auto start = Clock::now();
        dist::EnqueueMeshWorkload(queue, workload, false);
        dist::Finish(queue);
        const auto end = Clock::now();

        // 출력 전송과 반복 결과 비교는 측정에 포함하지 않습니다.
        actual.clear();
        dist::EnqueueReadMeshBuffer(queue, actual, output, true);
        if (i == 0) {
            reference_output = actual;
        } else if (actual != reference_output) {
            throw std::runtime_error("Benchmark output changed between repetitions");
        }

        const double elapsed = milliseconds(start, end);
        if (i < 2)
            warmup_ms.push_back(elapsed);
        else
            samples_ms.push_back(elapsed);
    }

    // 마지막 출력은 host의 기존 CPU/padding 검사로도 검증됩니다.
    return {{"clock", "host std::chrono::steady_clock"},
            {"scope", "EnqueueMeshWorkload + Finish; simulator/host dispatch wall time"},
            {"warmup_ms", warmup_ms},
            {"samples_ms", samples_ms},
            {"warmup_count", 2},
            {"sample_count", iterations},
            {"all_repetition_outputs_equal", true},
            {"first_dispatch_is_not_guaranteed_cold_JIT", true},
            {"physical_NPU_latency", false}};
}

} // namespace af::benchmark
