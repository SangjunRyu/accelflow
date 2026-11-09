// TensorAccessor and CB APIs match pinned official eltwise_binary.
#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// DRAM 입력 두 타일을 L1 CB0/CB1로 공급하는 생산자입니다.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t count = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer a_cb(tt::CBIndex::c_0), b_cb(tt::CBIndex::c_1);
    const uint32_t bytes = get_tile_size(tt::CBIndex::c_0);
    for (uint32_t tile = 0; tile < count; ++tile) {
        // 빈 공간 확보 → DMA 전송 → 완료 확인 → 공개 순서로 compute와 동기화합니다.
        a_cb.reserve_back(1);
        b_cb.reserve_back(1);
        noc.async_read(a, a_cb, bytes, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read(b, b_cb, bytes, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        a_cb.push_back(1);
        b_cb.push_back(1);
    }
}
