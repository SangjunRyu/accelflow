#pragma once

// All shards process the same row groups, with disjoint contiguous tile columns.
static af::dist::MeshWorkload make_column_sharded(
    af::DeviceSession& session,
    const std::shared_ptr<af::dist::MeshBuffer>& xd,
    const std::shared_ptr<af::dist::MeshBuffer>& gd,
    const std::shared_ptr<af::dist::MeshBuffer>& yd,
    uint32_t ht, uint32_t wt, uint32_t w, float epsilon,
    uint32_t depth, uint32_t peers) {
    auto program = af::CreateProgram();
    const tt::tt_metal::CoreRange cores({0, 0}, {peers - 1, 0});

    // Shared allocation fixes mailbox and semaphore offsets across all workers.
    for (uint32_t cb : {0U, 1U, 2U, 4U, 5U, 16U, 26U, 27U, 28U}) {
        const uint32_t slots = cb >= 26 ? 1 : depth;
        af::CreateCircularBuffer(program, cores,
            af::CircularBufferConfig(slots * 4096, {{cb, af::DataFormat::Float32}})
                .set_page_size(cb, 4096));
    }
    const uint32_t ready = af::CreateSemaphore(program, cores, 0);
    const uint32_t valid = af::CreateSemaphore(program, cores, 0);
    std::vector<af::CoreCoord> physical;
    for (uint32_t rank = 0; rank < peers; ++rank)
        physical.push_back(session.device->worker_core_from_logical_core({rank, 0}));

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
    auto rk = af::CreateKernel(program, dir + "/column_reader.cpp", cores, reader);
    auto ck = af::CreateKernel(program, dir + "/column_compute.cpp", cores, math);
    auto wk = af::CreateKernel(program, dir + "/column_writer.cpp", cores, writer);

    uint32_t start = 0;
    for (uint32_t rank = 0; rank < peers; ++rank) {
        const af::CoreCoord core{rank, 0};
        const uint32_t count = wt / peers + (rank < wt % peers ? 1 : 0);
        for (uint32_t cb : {24U, 25U})
            af::CreateCircularBuffer(program, core,
                af::CircularBufferConfig(count * 4096, {{cb, af::DataFormat::Float32}})
                    .set_page_size(cb, 4096));
        std::vector<uint32_t> args{af::address32(xd->address()), af::address32(gd->address()),
            ht, wt, start, count, rank, peers, ready, valid};
        for (const auto& p : physical) { args.push_back(p.x); args.push_back(p.y); }
        af::SetRuntimeArgs(program, rk, core, args);
        af::SetRuntimeArgs(program, ck, core,
            {ht, count, std::bit_cast<uint32_t>(1.0f / w), std::bit_cast<uint32_t>(epsilon), rank, peers});
        af::SetRuntimeArgs(program, wk, core, {af::address32(yd->address()), ht, wt, start, count});
        start += count;
    }
    af::dist::MeshWorkload workload;
    workload.add_program(af::dist::MeshCoordinateRange(session.device->shape()), std::move(program));
    return workload;
}
