#pragma once
#include "../vector_add/common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>

namespace af::mcast {

// 논리 코어 한 행에 C의 열을 분배하고 A만 NoC multicast로 공유합니다.
inline nlohmann::json configure(DeviceSession& session, Program& program,
                                const std::shared_ptr<dist::MeshBuffer>& ad,
                                const std::shared_ptr<dist::MeshBuffer>& bd,
                                const std::shared_ptr<dist::MeshBuffer>& cd,
                                uint32_t mt, uint32_t kt, uint32_t nt,
                                uint32_t depth, uint32_t requested) {
    const auto grid = session.device->compute_with_storage_grid_size();
    const uint32_t peers = std::min(requested, nt);
    if (peers == 0 || peers > grid.x)
        throw std::runtime_error("A multicast requires a nonempty single logical core row");

    std::vector<CoreCoord> physical;
    for (uint32_t rank = 0; rank < peers; ++rank)
        physical.push_back(session.device->worker_core_from_logical_core({rank, 0}));
    // Multicast 사각형이 불필요한 코어나 DRAM/Ethernet을 포함하지 않는지 확인합니다.
    for (uint32_t rank = 1; rank < peers; ++rank) {
        if (physical[rank].y != physical[0].y || physical[rank].x != physical[0].x + rank)
            throw std::runtime_error("Selected physical core row is not contiguous for multicast");
    }
    const tt::tt_metal::CoreRange cores({0,0},{peers-1,0});
    for (auto cb : {CBIndex::c_0,CBIndex::c_1,CBIndex::c_16}) {
        CreateCircularBuffer(program,cores,
            CircularBufferConfig(depth*2048,{{cb,DataFormat::Float16_b}}).set_page_size(cb,2048));
    }
    // 같은 코어 집합에 함께 할당해 CB와 semaphore의 L1 offset을 맞춥니다.
    const uint32_t ready = CreateSemaphore(program,cores,0);
    const uint32_t valid = CreateSemaphore(program,cores,0);
    DataMovementConfig read_config{},write_config{};
    read_config.processor = DataMovementProcessor::RISCV_0;
    read_config.noc = NOC::RISCV_0_default;
    TensorAccessorArgs(*ad).append_to(read_config.compile_args);
    TensorAccessorArgs(*bd).append_to(read_config.compile_args);
    write_config.processor = DataMovementProcessor::RISCV_1;
    write_config.noc = NOC::RISCV_1_default;
    TensorAccessorArgs(*cd).append_to(write_config.compile_args);
    ComputeConfig math_config{};
    math_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    math_config.fp32_dest_acc_en = true;
    const std::string kernels = ACCELFLOW_KERNEL_DIR;
    KernelHandle reader = CreateKernel(program,kernels+"/reader_mcast.cpp",cores,read_config);
    KernelHandle compute = CreateKernel(program,kernels+"/compute_mcast.cpp",cores,math_config);
    KernelHandle writer = CreateKernel(program,kernels+"/writer_mcast.cpp",cores,write_config);

    const auto sender = physical[0];
    const auto first = physical[peers>1 ? 1 : 0];
    const auto last = physical.back();
    nlohmann::json assignments = nlohmann::json::array();
    for (uint32_t rank = 0; rank < peers; ++rank) {
        const CoreCoord core{rank,0};
        SetRuntimeArgs(program,reader,core,
            {address32(ad->address()),address32(bd->address()),mt,kt,nt,rank,peers,
             uint32_t(sender.x),uint32_t(sender.y),uint32_t(first.x),uint32_t(first.y),
             uint32_t(last.x),uint32_t(last.y),ready,valid});
        SetRuntimeArgs(program,compute,core,{mt,kt,nt,rank,peers});
        SetRuntimeArgs(program,writer,core,{address32(cd->address()),mt,nt,rank,peers});
        std::vector<uint32_t> outputs;
        for (uint32_t m=0;m<mt;++m)
            for (uint32_t n=rank;n<nt;n+=peers)
                outputs.push_back(m*nt+n);
        assignments.push_back({{"logical_core",{rank,0}},
            {"physical_core",{physical[rank].x,physical[rank].y}},
            {"role",rank==0 ? "A sender + compute" : "A receiver + compute"},
            {"output_tile_ids",outputs}});
    }
    const uint64_t reads = uint64_t(mt)*kt*((nt+peers-1)/peers);
    return {{"grid",{grid.x,grid.y}},{"requested_cores",requested},{"active_cores",peers},
            {"mapping","strided output columns; A multicast on one core row"},
            {"assignments",assignments},{"a_dram_reads_model",reads},
            {"a_multicast_calls_model",peers>1 ? reads : 0},
            {"a_receiver_deliveries_model",reads*(peers-1)},
            {"unused_tail_a_tiles_model",uint64_t(mt)*kt*(peers*((nt+peers-1)/peers)-nt)}};
}

} // namespace af::mcast
