#pragma once
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace af {
using namespace tt;
using namespace tt::tt_metal;
namespace dist = tt::tt_metal::distributed;

// 공유 장치 수명 관리: 정상 종료는 close(), 예외 경로는 소멸자가 정리합니다.
struct DeviceSession {
    std::shared_ptr<dist::MeshDevice> device = dist::MeshDevice::create_unit_mesh(0);
    bool closed = false;

    void close() {
        const bool ok = device->close();
        closed = true;
        if (!ok)
            throw std::runtime_error("Device close failed");
    }

    ~DeviceSession() {
        if (!closed) {
            try {
                device->close();
            } catch (...) {
                std::cerr << "Device cleanup failed\n";
            }
        }
    }
};

// 장치 buffer를 생성합니다. bytes는 논리 크기이며 실제 정렬 크기는 따로 관찰합니다.
inline auto buffer(DeviceSession& session, uint32_t bytes, uint32_t page, BufferType type) {
    dist::DeviceLocalBufferConfig local{};
    local.page_size = page;
    local.buffer_type = type;
    dist::ReplicatedBufferConfig replicated{.size = bytes};
    return dist::MeshBuffer::create(replicated, local, session.device.get());
}

// 커널 runtime argument의 32-bit 주소 범위를 확인해 잘림을 방지합니다.
inline uint32_t address32(uint64_t address) {
    if (address > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Device buffer address exceeds the 32-bit kernel ABI");
    return static_cast<uint32_t>(address);
}

// 논리/page/정렬 크기를 기록합니다. 전체 allocator 메모리 사용량을 뜻하지 않습니다.
inline nlohmann::json buffer_observation(const std::shared_ptr<dist::MeshBuffer>& buffer) {
    const auto* device_buffer = buffer->get_reference_buffer();
    return {{"logical_bytes", buffer->device_local_size()},
            {"page_bytes", buffer->page_size()},
            {"pages", buffer->num_pages()},
            {"aligned_page_bytes", device_buffer->aligned_page_size()},
            {"aligned_size_bytes", device_buffer->aligned_size()},
            {"aligned_size_per_bank_bytes", device_buffer->aligned_size_per_bank()}};
}

template <typename T> uint64_t input_fingerprint(const std::vector<T>& input) {
    // FNV-1a is only an input-equality check, not a cryptographic provenance hash.
    uint64_t value = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(input.data());
    for (size_t i = 0; i < input.size() * sizeof(T); ++i) {
        value ^= bytes[i];
        value *= 1099511628211ULL;
    }
    return value;
}

// 프로그램을 mesh workload로 제출하고 장치 완료까지 기다립니다.
inline void execute(DeviceSession& session, Program&& program) {
    dist::MeshWorkload workload;
    workload.add_program(dist::MeshCoordinateRange(session.device->shape()), std::move(program));
    dist::EnqueueMeshWorkload(session.device->mesh_command_queue(), workload, false);
    dist::Finish(session.device->mesh_command_queue());
}

// 검증 결과를 읽기 쉬운 JSON으로 저장하고 쓰기 실패도 오류로 처리합니다.
inline void save(const char* path, const nlohmann::json& report) {
    std::ofstream stream(path);
    stream << report.dump(2) << '\n';
    if (!stream)
        throw std::runtime_error("Could not write validation report");
}
} // namespace af
