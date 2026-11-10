#pragma once
#include "../vector_add/common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>

namespace af::mcast2d {

// requested=4는 기본 2×2, 한 축이 한 타일뿐이면 1×4 또는 4×1로 배치합니다.
inline nlohmann::json configure(DeviceSession& session, Program& program,
                                const std::shared_ptr<dist::MeshBuffer>& ad,
                                const std::shared_ptr<dist::MeshBuffer>& bd,
                                const std::shared_ptr<dist::MeshBuffer>& cd,
                                uint32_t mt, uint32_t kt, uint32_t nt,
                                uint32_t depth, uint32_t requested, uint32_t batch = 1, bool batched = false) {
    uint32_t rows = 1, cols = 1;
    if (requested == 2) {
        if (nt > 1)
            cols = std::min(2U, nt);
        else
            rows = std::min(2U, mt);
    } else if (requested == 4) {
        if (mt == 1)
            cols = std::min(4U, nt);
        else if (nt == 1)
            rows = std::min(4U, mt);
        else {
            rows = std::min(2U, mt);
            cols = std::min(2U, nt);
        }
    } else if (requested != 1) {
        throw std::runtime_error("2D multicast supports requests 1, 2, 4");
    }

    const auto grid = session.device->compute_with_storage_grid_size();
    if (rows > grid.y || cols > grid.x)
        throw std::runtime_error("2D multicast grid exceeds available cores");
    const uint32_t peers = rows * cols;
    std::vector<CoreCoord> physical;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            physical.push_back(session.device->worker_core_from_logical_core({c,r}));
            const auto p = physical.back();
            if (p.x != physical[0].x + c || p.y != physical[0].y + r)
                throw std::runtime_error("2D multicast requires contiguous physical worker rows/columns");
        }
    }
    const tt::tt_metal::CoreRange cores({0,0},{cols-1,rows-1});
    for (auto cb : {CBIndex::c_0,CBIndex::c_1,CBIndex::c_16}) {
        CreateCircularBuffer(program,cores,
            CircularBufferConfig(depth*(cb == CBIndex::c_16 ? 1 : batch)*2048,{{cb,DataFormat::Float16_b}}).set_page_size(cb,2048));
    }
    // A 행과 B 열의 준비/도착 신호는 독립된 semaphore 쌍을 사용합니다.
    const uint32_t a_ready = CreateSemaphore(program,cores,0);
    const uint32_t a_valid = CreateSemaphore(program,cores,0);
    const uint32_t b_ready = CreateSemaphore(program,cores,0);
    const uint32_t b_valid = CreateSemaphore(program,cores,0);

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
    KernelHandle reader = CreateKernel(program,kernels+(batched ? "/reader_mcast2d_batched.cpp" : "/reader_mcast2d.cpp"),cores,read_config);
    KernelHandle compute = CreateKernel(program,kernels+(batched ? "/compute_mcast2d_batched.cpp" : "/compute_mcast2d.cpp"),cores,math_config);
    KernelHandle writer = CreateKernel(program,kernels+"/writer_mcast2d.cpp",cores,write_config);

    nlohmann::json assignments = nlohmann::json::array();
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const CoreCoord core{c,r};
            const auto a_sender = physical[r*cols];
            const auto a_first = physical[r*cols+(cols>1 ? 1 : 0)];
            const auto a_last = physical[r*cols+cols-1];
            const auto b_sender = physical[c];
            const auto b_first = physical[(rows>1 ? cols : 0)+c];
            const auto b_last = physical[(rows-1)*cols+c];
            SetRuntimeArgs(program,reader,core,
                {address32(ad->address()),address32(bd->address()),mt,kt,nt,r,c,rows,cols,
                 uint32_t(a_sender.x),uint32_t(a_sender.y),uint32_t(a_first.x),uint32_t(a_first.y),
                 uint32_t(a_last.x),uint32_t(a_last.y),a_ready,a_valid,
                 uint32_t(b_sender.x),uint32_t(b_sender.y),uint32_t(b_first.x),uint32_t(b_first.y),
                 uint32_t(b_last.x),uint32_t(b_last.y),b_ready,b_valid,batch});
            SetRuntimeArgs(program,compute,core,{mt,kt,nt,r,c,rows,cols,batch});
            SetRuntimeArgs(program,writer,core,{address32(cd->address()),mt,nt,r,c,rows,cols});

            std::vector<uint32_t> outputs;
            for (uint32_t m = r; m < mt; m += rows)
                for (uint32_t n = c; n < nt; n += cols)
                    outputs.push_back(m*nt+n);
            const auto p = physical[r*cols+c];
            assignments.push_back({{"logical_core",{c,r}},{"physical_core",{p.x,p.y}},
                {"A_sender",c==0},{"B_sender",r==0},{"output_tile_ids",outputs}});
        }
    }
    const uint32_t mg = (mt+rows-1)/rows, ng = (nt+cols-1)/cols;
    const uint64_t a_reads = uint64_t(mt)*kt*ng;
    const uint64_t b_reads = uint64_t(nt)*kt*mg;
    const uint64_t chunks = (kt+batch-1)/batch;
    const uint64_t a_batches = uint64_t(mt)*chunks*ng;
    const uint64_t b_batches = uint64_t(nt)*chunks*mg;
    return {{"batch_tiles",batch},{"read_barriers_model",a_batches+b_batches},
            {"ready_atomic_increments_model",a_batches*(cols-1)+b_batches*(rows-1)},
            {"grid",{grid.x,grid.y}},{"core_grid_rows_cols",{rows,cols}},
            {"requested_cores",requested},{"active_cores",peers},{"assignments",assignments},
            {"a_dram_reads_model",a_reads},{"b_dram_reads_model",b_reads},
            {"a_multicast_calls_model",cols>1 ? a_batches : 0},
            {"b_multicast_calls_model",rows>1 ? b_batches : 0},
            {"a_receiver_deliveries_model",a_reads*(cols-1)},
            {"b_receiver_deliveries_model",b_reads*(rows-1)},
            {"unused_tail_a_tiles_model",uint64_t(mt)*kt*(ng*cols-nt)},
            {"unused_tail_b_tiles_model",uint64_t(nt)*kt*(mg*rows-mt)},
            {"dummy_a_slots_model",uint64_t(mg*rows-mt)*ng*cols*kt},
            {"dummy_b_slots_model",uint64_t(ng*cols-nt)*mg*rows*kt},
            {"cb_phases_per_core_model",uint64_t(mg)*ng*chunks}};
}

} // namespace af::mcast2d
