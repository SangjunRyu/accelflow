#pragma once
#include <chrono>
#include <string>
#include <vector>

namespace af::rms_bench {
struct Options {
    bool enabled = false;
    uint32_t warmups = 0;
    uint32_t samples = 0;
    uint32_t iterations(uint32_t normal_repeats) const {
        return enabled ? warmups + samples : normal_repeats;
    }
};
inline Options options;

// Both hosts keep their original seven-argument CLI and accept this same suffix.
inline int configure(int argc, char** argv) {
    if (argc != 10) return argc;
    if (std::string(argv[7]) != "--benchmark")
        throw std::runtime_error("Expected --benchmark warmups samples");
    options = {true, uint32_t(std::stoul(argv[8])), uint32_t(std::stoul(argv[9]))};
    if (options.warmups < 1 || options.warmups > 3 || options.samples < 3 || options.samples > 11)
        throw std::runtime_error("Benchmark requires 1..3 warmups and 3..11 samples");
    return 7;
}

class PipelineTimer {
    using Clock = std::chrono::steady_clock;
    std::vector<double> warmups_, samples_;
    std::vector<std::vector<double>> stage_warmups_, stage_samples_;
    static double ms(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    }
public:
    template<class Workloads, class Queue>
    void submit(Workloads& workloads, Queue& queue, uint32_t iteration) {
        if (!options.enabled) {
            for (auto& work : workloads) {
                dist::EnqueueMeshWorkload(queue,work,false);
                dist::Finish(queue);
            }
            return;
        }
        // Input/sentinel writes must finish before starting the host clock.
        dist::Finish(queue);
        std::vector<double> stage;
        stage.reserve(workloads.size());
        const auto begin=Clock::now();
        for (auto& work : workloads) {
            const auto a=Clock::now();
            dist::EnqueueMeshWorkload(queue,work,false);
            dist::Finish(queue);
            const auto b=Clock::now();
            stage.push_back(ms(a,b));
        }
        const auto end=Clock::now();
        if (iteration < options.warmups) {
            warmups_.push_back(ms(begin,end));
            stage_warmups_.push_back(std::move(stage));
        } else {
            samples_.push_back(ms(begin,end));
            stage_samples_.push_back(std::move(stage));
        }
    }

    nlohmann::json json() const {
        return {{"enabled",options.enabled},{"clock","host steady_clock"},
            {"scope","3-stage enqueue/Finish pipeline including runtime and simulator wait"},
            {"physical_npu_latency",false},{"warmup_count",options.warmups},{"sample_count",options.samples},
            {"warmup_ms",warmups_},{"samples_ms",samples_},
            {"stage_warmup_ms",stage_warmups_},{"stage_samples_ms",stage_samples_},
            {"stage_names",{"square","sum","normalize"}},
            {"excluded","initialization, allocation, fixture packing, input/sentinel writes, readback and CPU validation"}};
    }
};
} // namespace af::rms_bench
