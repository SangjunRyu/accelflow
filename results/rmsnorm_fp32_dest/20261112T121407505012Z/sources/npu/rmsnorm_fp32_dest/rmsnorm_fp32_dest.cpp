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
    uint32_t depth, bool compat = false) {
    auto program = af::CreateProgram();
    const af::CoreCoord core{0, 0};
    for (uint32_t cb : {0U,3U,16U})
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
    auto math = compute_config(direct,{0});
    if (compat) math.defines["ACCELFLOW_FP32_DUMMY_WORKAROUND"] = "1";
    auto ck = af::CreateKernel(program,dir+"/"+kernel+".cpp",core,math);
    auto wk = af::CreateKernel(program,dir+"/writer.cpp",core,writer);
    af::SetRuntimeArgs(program,rk,core,{af::address32(input->address()),input_tiles});
    af::SetRuntimeArgs(program,ck,core,math_args);
    af::SetRuntimeArgs(program,wk,core,{af::address32(output->address()),output_tiles});
    af::dist::MeshWorkload work;
    work.add_program(af::dist::MeshCoordinateRange(session.device->shape()),std::move(program));
    return work;
}

static int probe(const std::string& report, bool direct, bool transpose, bool compat) {
    af::DeviceSession session;
    auto input = af::buffer(session,16384,4096,af::BufferType::DRAM);
    auto output = af::buffer(session,16384,4096,af::BufferType::DRAM);
    std::vector<float> values(4096);
    for (uint32_t i=0;i<4096;++i) {
        uint32_t bits=0x3f800001U + (i*37)%8191;
        if (i%3==0) bits |= 0x80000000U;
        values[i]=std::bit_cast<float>(bits);
    }
    const auto packed=pack(values,128,32);
    auto& queue=session.device->mesh_command_queue();
    af::dist::EnqueueWriteMeshBuffer(queue,input,packed,false);
    af::dist::EnqueueWriteMeshBuffer(queue,output,std::vector<float>(4096,std::numeric_limits<float>::quiet_NaN()),false);
    auto work=make_stage(session,direct,"probe",input,output,4,4,{transpose?1U:0U,4},1,compat);
    af::dist::EnqueueMeshWorkload(queue,work,false);
    af::dist::Finish(queue);
    std::vector<float> actual;
    af::dist::EnqueueReadMeshBuffer(queue,actual,output,true);
    uint32_t mismatches=0;double max_error=0;
    nlohmann::json examples=nlohmann::json::array();
    for(uint32_t r=0;r<128;++r) for(uint32_t c=0;c<32;++c) {
        const float expected=values[transpose ? ((r/32)*32+c)*32+r%32 : r*32+c];
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
    af::save(report.c_str(),{{"passed",passed},{"test","transport"},{"mode",compat?"compat":direct?"direct":"default"},
        {"operation",transpose?"transpose":"copy"},{"values_checked",4096},{"bit_mismatches",mismatches},
        {"max_abs_error",max_error},{"negative_control",!direct},{"examples",examples}});
    std::cout << "PROBE " << (direct?"direct":"default") << " mismatches=" << mismatches << std::endl;
    return passed?0:1;
}


static af::dist::MeshWorkload make_normalize(
    af::DeviceSession& session, bool direct, bool compat,
    const std::shared_ptr<af::dist::MeshBuffer>& x,
    const std::shared_ptr<af::dist::MeshBuffer>& gamma,
    const std::shared_ptr<af::dist::MeshBuffer>& sums,
    const std::shared_ptr<af::dist::MeshBuffer>& output,
    uint32_t ht, uint32_t wt, uint32_t w, float epsilon, uint32_t depth) {
    auto program=af::CreateProgram();
    const af::CoreCoord core{0,0};
    for (uint32_t cb : {0U,1U,2U,3U,16U}) {
        const uint32_t slots=cb==3?1:depth;
        af::CreateCircularBuffer(program,core,
            af::CircularBufferConfig(slots*4096,{{cb,af::DataFormat::Float32}}).set_page_size(cb,4096));
    }
    af::DataMovementConfig reader{},writer{};
    reader.processor=af::DataMovementProcessor::RISCV_0;
    reader.noc=af::NOC::RISCV_0_default;
    writer.processor=af::DataMovementProcessor::RISCV_1;
    writer.noc=af::NOC::RISCV_1_default;
    af::TensorAccessorArgs(*x).append_to(reader.compile_args);
    af::TensorAccessorArgs(*gamma).append_to(reader.compile_args);
    af::TensorAccessorArgs(*sums).append_to(reader.compile_args);
    af::TensorAccessorArgs(*output).append_to(writer.compile_args);
    auto math=compute_config(direct,{0,1,2});
    if(compat) math.defines["ACCELFLOW_FP32_DUMMY_WORKAROUND"]="1";
    const std::string dir=ACCELFLOW_KERNEL_DIR;
    auto rk=af::CreateKernel(program,dir+"/reader_normalize.cpp",core,reader);
    auto ck=af::CreateKernel(program,dir+"/normalize.cpp",core,math);
    auto wk=af::CreateKernel(program,dir+"/writer.cpp",core,writer);
    af::SetRuntimeArgs(program,rk,core,{af::address32(x->address()),af::address32(gamma->address()),
        af::address32(sums->address()),ht,wt});
    af::SetRuntimeArgs(program,ck,core,{ht*wt,std::bit_cast<uint32_t>(1.0f/w),std::bit_cast<uint32_t>(epsilon)});
    af::SetRuntimeArgs(program,wk,core,{af::address32(output->address()),ht*wt});
    af::dist::MeshWorkload work;
    work.add_program(af::dist::MeshCoordinateRange(session.device->shape()),std::move(program));
    return work;
}

// AFRM0001 retains the old BF16 fixtures; AFRF0001 contains actual FP32 words.
static int run_rmsnorm(const char* list_path,const char* report_path,uint32_t depth,uint32_t repeats,
                       bool direct,bool compat) {
    if((depth!=1 && depth!=2) || repeats<1 || repeats>3)
        throw std::runtime_error("Invalid depth/repeats");
    std::ifstream list(list_path);
    if(!list) throw std::runtime_error("Cannot open fixture list");
    af::DeviceSession session;
    auto& queue=session.device->mesh_command_queue();
    nlohmann::json cases=nlohmann::json::array();
    bool all_passed=true;
    std::string path;
    while(std::getline(list,path)) {
        if(path.empty()) continue;
        std::ifstream file(path,std::ios::binary);
        char magic[8];uint32_t dims[4]{};float epsilon=0;
        file.read(magic,8);file.read(reinterpret_cast<char*>(dims),16);
        file.read(reinterpret_cast<char*>(&epsilon),4);
        const uint32_t h=dims[0],w=dims[1],hp=dims[2],wp=dims[3];
        const bool bf16=std::string(magic,8)=="AFRM0001";
        if(!file || (!bf16 && std::string(magic,8)!="AFRF0001") || !h || !w || h>2048 || w>4097 ||
           hp!=(h+31)/32*32 || wp!=(w+31)/32*32 || !std::isfinite(epsilon) || epsilon<=0)
            throw std::runtime_error("Invalid fixture header");
        std::vector<float> data(size_t(hp)*wp+wp);
        if(bf16) {
            std::vector<uint16_t> bits(data.size());
            file.read(reinterpret_cast<char*>(bits.data()),bits.size()*2);
            for(size_t i=0;i<bits.size();++i) data[i]=std::bit_cast<float>(uint32_t(bits[i])<<16);
        } else file.read(reinterpret_cast<char*>(data.data()),data.size()*4);
        if(!file || file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Fixture size mismatch");
        if(std::any_of(data.begin(),data.end(),[](float v){return !std::isfinite(v);}))
            throw std::runtime_error("Only finite input fixtures are supported");
        const std::vector<float> x(data.begin(),data.begin()+size_t(hp)*wp);
        std::vector<float> gamma(32*wp);
        for(uint32_t r=0;r<32;++r) for(uint32_t c=0;c<wp;++c) gamma[r*wp+c]=data[size_t(hp)*wp+c];
        const uint32_t ht=hp/32,wt=wp/32,tiles=ht*wt;
        auto xd=af::buffer(session,tiles*4096,4096,af::BufferType::DRAM);
        auto gd=af::buffer(session,wt*4096,4096,af::BufferType::DRAM);
        auto squared=af::buffer(session,tiles*4096,4096,af::BufferType::DRAM);
        auto sums=af::buffer(session,ht*4096,4096,af::BufferType::DRAM);
        auto output=af::buffer(session,tiles*4096,4096,af::BufferType::DRAM);
        const auto px=pack(x,hp,wp),pg=pack(gamma,32,wp);
        af::dist::EnqueueWriteMeshBuffer(queue,xd,px,false);
        af::dist::EnqueueWriteMeshBuffer(queue,gd,pg,false);
        std::vector<af::dist::MeshWorkload> stages;
        stages.push_back(make_stage(session,direct,"square",xd,squared,tiles,tiles,{tiles},depth,compat));
        stages.push_back(make_stage(session,direct,"sum",squared,sums,tiles,ht,{ht,wt},depth,compat));
        stages.push_back(make_normalize(session,direct,compat,xd,gd,sums,output,ht,wt,w,epsilon,depth));
        std::vector<double> reference(size_t(hp)*wp,0),sum_reference(hp,0);
        for(uint32_t r=0;r<h;++r) {
            for(uint32_t c=0;c<w;++c) sum_reference[r]+=double(x[r*wp+c])*x[r*wp+c];
            const double inv=1.0/std::sqrt(sum_reference[r]/w+double(epsilon));
            for(uint32_t c=0;c<w;++c) reference[r*wp+c]=double(x[r*wp+c])*inv*gamma[c];
        }
        std::vector<float> first,first_sums;
        uint32_t bad_output=0,bad_sums=0,square_bit_differences=0;
        double max_error=0,max_sum_error=0,max_square_error=0;
        for(uint32_t repeat=0;repeat<repeats;++repeat) {
            const std::vector<float> sentinel(size_t(hp)*wp,std::numeric_limits<float>::quiet_NaN());
            af::dist::EnqueueWriteMeshBuffer(queue,squared,sentinel,false);
            af::dist::EnqueueWriteMeshBuffer(queue,output,sentinel,false);
            af::dist::EnqueueWriteMeshBuffer(queue,sums,std::vector<float>(ht*1024,std::numeric_limits<float>::quiet_NaN()),false);
            for(auto& stage:stages) {
                af::dist::EnqueueMeshWorkload(queue,stage,false);
                af::dist::Finish(queue);
            }
            std::vector<float> actual,sq,stat;
            af::dist::EnqueueReadMeshBuffer(queue,actual,output,true);
            af::dist::EnqueueReadMeshBuffer(queue,sq,squared,true);
            af::dist::EnqueueReadMeshBuffer(queue,stat,sums,true);
            if(first.empty()) { first=actual;first_sums=stat; }
            else if(std::memcmp(first.data(),actual.data(),first.size()*4)!=0 ||
                    std::memcmp(first_sums.data(),stat.data(),first_sums.size()*4)!=0)
                throw std::runtime_error("Repeated results differ bitwise");
            for(uint32_t r=0;r<hp;++r) {
                const float got_sum=stat[(r/32)*1024+offset(31,r%32,32)];
                const double err_sum=std::abs(double(got_sum)-sum_reference[r]);
                if(!std::isfinite(got_sum) || err_sum>1e-4+1e-4*std::abs(sum_reference[r])) ++bad_sums;
                max_sum_error=std::max(max_sum_error,err_sum);
                for(uint32_t c=0;c<wp;++c) {
                    const size_t i=offset(r,c,wp),logical=r*wp+c;
                    const float expected_square=x[logical]*x[logical];
                    const double error=std::abs(double(actual[i])-reference[logical]);
                    if(!std::isfinite(actual[i]) || error>1e-4+1e-4*std::abs(reference[logical]) ||
                       ((r>=h || c>=w) && actual[i]!=0)) ++bad_output;
                    if(!std::isfinite(sq[i])) throw std::runtime_error("Non-finite square stage");
                    if(std::bit_cast<uint32_t>(sq[i])!=std::bit_cast<uint32_t>(expected_square)) ++square_bit_differences;
                    max_square_error=std::max(max_square_error,std::abs(double(sq[i])-expected_square));
                    max_error=std::max(max_error,error);
                }
            }
        }
        std::vector<float> logical_output,logical_sums;
        for(uint32_t r=0;r<h;++r) {
            logical_sums.push_back(first_sums[(r/32)*1024+offset(31,r%32,32)]);
            for(uint32_t c=0;c<w;++c) logical_output.push_back(first[offset(r,c,wp)]);
        }
        const bool passed=bad_output==0 && bad_sums==0;
        all_passed &= passed;
        cases.push_back({{"key",std::filesystem::path(path).stem().string()},{"shape_hw",{h,w}},
            {"fixture_dtype",bf16?"BF16 widened to FP32":"FP32"},{"epsilon",epsilon},{"passed",passed},
            {"logical_output",logical_output},{"logical_sum_squares",logical_sums},{"repeats",repeats},
            {"bad_output_values",bad_output},{"bad_sum_rows",bad_sums},
            {"max_abs_error",max_error},{"max_sum_square_error",max_sum_error},
            {"square_bit_differences",square_bit_differences},{"max_square_abs_error",max_square_error},
            {"padding_zero",bad_output==0},{"repeats_bitwise_equal",true},
            {"input_fingerprint",af::input_fingerprint(px)},{"gamma_fingerprint",af::input_fingerprint(pg)},
            {"output_fingerprint",af::input_fingerprint(first)},
            {"cb_peak_payload_bytes_model",(4*depth+1)*4096},
            {"intermediate_dram_bytes_model",uint64_t(tiles+ht)*4096}});
        std::cout<<(passed?"PASS ":"FAIL ")<<path<<" max_abs="<<max_error<<" sum_error="<<max_sum_error<<std::endl;
    }
    if(cases.empty()) throw std::runtime_error("No fixtures");
    session.close();
    af::save(report_path,{{"passed",all_passed},{"mode",compat?"compat":direct?"direct":"default"},
        {"fp32_dest_acc_en",true},{"unpack_to_dest_fp32",direct},{"local_dummy_unpack_workaround",compat},
        {"residual_bit_splitting",false},{"core_count",1},{"stages",3},{"depth",depth},
        {"atol",1e-4},{"rtol",1e-4},{"cases",cases}});
    return all_passed?0:1;
}

int main(int argc,char** argv) {
    try {
        if(argc==5 && std::string(argv[1])=="probe") {
            const std::string mode=argv[3],op=argv[4];
            if((mode!="default" && mode!="direct" && mode!="compat") || (op!="copy" && op!="transpose"))
                throw std::runtime_error("Invalid probe mode/operation");
            return probe(argv[2],mode!="default",op=="transpose",mode=="compat");
        }
        if(argc==7 && std::string(argv[1])=="run") {
            const std::string mode=argv[6];
            if(mode!="default" && mode!="direct" && mode!="compat") throw std::runtime_error("Invalid RMSNorm mode");
            return run_rmsnorm(argv[2],argv[3],std::stoul(argv[4]),std::stoul(argv[5]),
                mode!="default",mode=="compat");
        }
        throw std::runtime_error("Usage: rmsnorm_fp32_dest probe report.json default|direct|compat copy|transpose; or run list.txt report.json depth repeats default|direct|compat");
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
