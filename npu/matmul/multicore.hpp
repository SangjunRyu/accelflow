#pragma once

#include "../vector_add/common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>

namespace af::multicore {

// C의 연속 타일 구간을 최대 requested개 코어에 균등 분배합니다.
// 나머지는 앞 코어부터 하나씩 더 배정하여 코어 간 타일 수 차이를 1 이하로 만듭니다.
inline nlohmann::json configure(DeviceSession& session,
                                Program& program,
                                const std::shared_ptr<dist::MeshBuffer>& ad,
                                const std::shared_ptr<dist::MeshBuffer>& bd,
                                const std::shared_ptr<dist::MeshBuffer>& cd,
                                uint32_t mt,
                                uint32_t kt,
                                uint32_t nt,
                                uint32_t depth,
                                uint32_t requested) {
    const auto grid = session.device->compute_with_storage_grid_size();
    if (requested == 0 || requested > grid.x * grid.y)
        throw std::runtime_error("Requested core count exceeds available compute grid");

    const uint32_t tiles = mt * nt;
    const uint32_t active = std::min(requested, tiles);
    const uint32_t quotient = tiles / active;
    const uint32_t remainder = tiles % active;
    const std::string kernels = ACCELFLOW_KERNEL_DIR;
    nlohmann::json assignments = nlohmann::json::array();
    uint32_t start = 0;

    for (uint32_t index = 0; index < active; ++index) {
        // 논리 코어를 row-major로 선택합니다. 물리 NoC 좌표 변환은 runtime에 맡깁니다.
        const CoreCoord core{index % grid.x, index / grid.x};
        const uint32_t count = quotient + (index < remainder ? 1 : 0);

        // 같은 CB ID를 사용해도 L1의 실제 버퍼는 코어마다 독립적입니다.
        for (auto cb : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
            CreateCircularBuffer(
                program, core,
                CircularBufferConfig(depth * 2048, {{cb, DataFormat::Float16_b}})
                    .set_page_size(cb, 2048));
        }

        DataMovementConfig reader_config{}, writer_config{};
        reader_config.processor = DataMovementProcessor::RISCV_1;
        reader_config.noc = NOC::RISCV_1_default;
        TensorAccessorArgs(*ad).append_to(reader_config.compile_args);
        TensorAccessorArgs(*bd).append_to(reader_config.compile_args);
        writer_config.processor = DataMovementProcessor::RISCV_0;
        writer_config.noc = NOC::RISCV_0_default;
        TensorAccessorArgs(*cd).append_to(writer_config.compile_args);

        ComputeConfig compute_config{};
        compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
        compute_config.fp32_dest_acc_en = true;

        KernelHandle reader = CreateKernel(
            program, kernels + "/reader_multicore.cpp", core, reader_config);
        KernelHandle compute = CreateKernel(
            program, kernels + "/compute_multicore.cpp", core, compute_config);
        KernelHandle writer = CreateKernel(
            program, kernels + "/writer_multicore.cpp", core, writer_config);

        SetRuntimeArgs(program, reader, core,
                       {address32(ad->address()), address32(bd->address()), kt, nt, start, count});
        SetRuntimeArgs(program, compute, core, {kt, count});
        SetRuntimeArgs(program, writer, core, {address32(cd->address()), start, count});

        assignments.push_back({{"logical_core", {core.x, core.y}},
                               {"start_tile", start}, {"tile_count", count}});
        start += count;
    }

    if (start != tiles)
        throw std::runtime_error("Multicore assignment does not cover output");
    return {{"grid", {grid.x, grid.y}}, {"requested_cores", requested},
            {"active_cores", active}, {"assignments", assignments}};
}

} // namespace af::multicore
