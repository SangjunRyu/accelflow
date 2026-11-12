#include "../vector_add/common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>

static size_t offset(uint32_t row, uint32_t col, uint32_t width) {
    const uint32_t r = row % 32, c = col % 32;
    return (row / 32 * (width / 32) + col / 32) * 1024 +
           (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16;
}

static std::vector<float> pack(const std::vector<float>& values, uint32_t h, uint32_t w) {
    std::vector<float> result(values.size());
    for (uint32_t r = 0; r < h; ++r)
        for (uint32_t c = 0; c < w; ++c)
            result[offset(r, c, w)] = values[r * w + c];
    return result;
}

// Only copy_tile/transpose_tile consume marked CBs; all arithmetic is SFPU on Dest.
static af::ComputeConfig compute_config(bool direct, const std::vector<uint32_t>& inputs) {
    af::ComputeConfig config{};
    config.math_fidelity = af::MathFidelity::HiFi4;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.unpack_to_dest_mode.assign(32, af::UnpackToDestMode::Default);
    if (direct)
        for (uint32_t cb : inputs)
            config.unpack_to_dest_mode[cb] = af::UnpackToDestMode::UnpackToDestFp32;
    return config;
}

static af::dist::MeshWorkload make_stage(
    af::DeviceSession& session, bool direct, const std::string& kernel,
    const std::shared_ptr<af::dist::MeshBuffer>& input,
    const std::shared_ptr<af::dist::MeshBuffer>& output,
    uint32_t input_tiles, uint32_t output_tiles, const std::vector<uint32_t>& math_args,
    uint32_t depth) {
    auto program = af::CreateProgram();
    const af::CoreCoord core{0, 0};
    for (uint32_t cb : {0U,16U})
        af::CreateCircularBuffer(program, core,
            af::CircularBufferConfig(depth*4096, {{cb, af::DataFormat::Float32}}).set_page_size(cb,4096));
    af::DataMovementConfig reader{}, writer{};
    reader.processor = af::DataMovementProcessor::RISCV_0;
    reader.noc = af::NOC::RISCV_0_default;
    writer.processor = af::DataMovementProcessor::RISCV_1;
    writer.noc = af::NOC::RISCV_1_default;
    af::TensorAccessorArgs(*input).append_to(reader.compile_args);
    af::TensorAccessorArgs(*output).append_to(writer.compile_args);
    const std::string dir = ACCELFLOW_KERNEL_DIR;
    auto rk = af::CreateKernel(program,dir+"/reader.cpp",core,reader);
    auto ck = af::CreateKernel(program,dir+"/"+kernel+".cpp",core,compute_config(direct,{0}));
    auto wk = af::CreateKernel(program,dir+"/writer.cpp",core,writer);
    af::SetRuntimeArgs(program,rk,core,{af::address32(input->address()),input_tiles});
    af::SetRuntimeArgs(program,ck,core,math_args);
    af::SetRuntimeArgs(program,wk,core,{af::address32(output->address()),output_tiles});
    af::dist::MeshWorkload work;
    work.add_program(af::dist::MeshCoordinateRange(session.device->shape()),std::move(program));
    return work;
}

static int probe(const std::string& report, bool direct, bool transpose) {
    af::DeviceSession session;
    auto input = af::buffer(session,4096,4096,af::BufferType::DRAM);
    auto output = af::buffer(session,4096,4096,af::BufferType::DRAM);
    std::vector<float> values(1024);
    for (uint32_t i=0;i<1024;++i) {
        uint32_t bits=0x3f800001U + (i*37)%8191;
        if (i%3==0) bits |= 0x80000000U;
        values[i]=std::bit_cast<float>(bits);
    }
    const auto packed=pack(values,32,32);
    auto& queue=session.device->mesh_command_queue();
    af::dist::EnqueueWriteMeshBuffer(queue,input,packed,false);
    af::dist::EnqueueWriteMeshBuffer(queue,output,std::vector<float>(1024,std::numeric_limits<float>::quiet_NaN()),false);
    auto work=make_stage(session,direct,"probe",input,output,1,1,{transpose?1U:0U},1);
    af::dist::EnqueueMeshWorkload(queue,work,false);
    af::dist::Finish(queue);
    std::vector<float> actual;
    af::dist::EnqueueReadMeshBuffer(queue,actual,output,true);
    uint32_t mismatches=0;double max_error=0;
    nlohmann::json examples=nlohmann::json::array();
    for(uint32_t r=0;r<32;++r) for(uint32_t c=0;c<32;++c) {
        const float expected=values[(transpose?c:r)*32+(transpose?r:c)];
        const float got=actual[offset(r,c,32)];
        if(!std::isfinite(got)) throw std::runtime_error("Probe produced non-finite output");
        max_error=std::max(max_error,std::abs(double(got)-expected));
        if(std::bit_cast<uint32_t>(expected)!=std::bit_cast<uint32_t>(got)) {
            ++mismatches;
            if(examples.size()<4) examples.push_back({{"row",r},{"col",c},
                {"expected_bits",std::bit_cast<uint32_t>(expected)},{"actual_bits",std::bit_cast<uint32_t>(got)}});
        }
    }
    const bool passed=direct ? mismatches==0 : mismatches>0;
    session.close();
    af::save(report,{{"passed",passed},{"test","transport"},{"mode",direct?"direct":"default"},
        {"operation",transpose?"transpose":"copy"},{"values_checked",1024},{"bit_mismatches",mismatches},
        {"max_abs_error",max_error},{"negative_control",!direct},{"examples",examples}});
    std::cout << "PROBE " << (direct?"direct":"default") << " mismatches=" << mismatches << std::endl;
    return passed?0:1;
}

int main(int argc,char** argv) {
    try {
        if(argc==5 && std::string(argv[1])=="probe") {
            const std::string mode=argv[3],op=argv[4];
            if((mode!="default" && mode!="direct") || (op!="copy" && op!="transpose"))
                throw std::runtime_error("Invalid probe mode/operation");
            return probe(argv[2],mode=="direct",op=="transpose");
        }
        throw std::runtime_error("Usage: rmsnorm_fp32_dest probe report.json default|direct copy|transpose");
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
