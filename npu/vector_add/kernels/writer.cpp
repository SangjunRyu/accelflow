#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// compute의 출력 CB16을 소비해 DRAM에 기록합니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t bytes = get_tile_size(tt::CBIndex::c_16);
    CircularBuffer cb(tt::CBIndex::c_16);
    Noc noc;
    for (uint32_t tile = 0; tile < count; ++tile) {
        // 출력 준비 대기 → DMA 쓰기 완료 확인 → pop. 완료 전에 공간을 재사용하면 안 됩니다.
        cb.wait_front(1);
        noc.async_write(cb, output, bytes, {}, {.page_id = tile});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
