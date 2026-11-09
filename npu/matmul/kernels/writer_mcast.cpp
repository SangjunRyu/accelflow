#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 각 rank가 담당하는 C의 열만 공유 DRAM의 전역 주소에 기록합니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t mt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t rank = get_arg_val<uint32_t>(3);
    const uint32_t peers = get_arg_val<uint32_t>(4);

    for (uint32_t m = 0; m < mt; ++m) {
        for (uint32_t n = rank; n < nt; n += peers) {
            cb_wait_front(16, 1);
            noc_async_write_page(m * nt + n, c, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
}
