#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// L1 출력 CB → DRAM C. 타일 순서는 host가 복원하는 tile row-major 순서입니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        // compute가 공개한 타일을 기다리고, DMA 쓰기가 끝난 뒤 CB 공간을 반환합니다.
        cb_wait_front(16, 1);
        noc_async_write_page(tile, c, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
