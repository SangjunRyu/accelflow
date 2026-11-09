#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 코어의 행/열 stride에 맞는 유효 C 타일만 기록합니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t mt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t row = get_arg_val<uint32_t>(3);
    const uint32_t col = get_arg_val<uint32_t>(4);
    const uint32_t rows = get_arg_val<uint32_t>(5);
    const uint32_t cols = get_arg_val<uint32_t>(6);
    for (uint32_t m = row; m < mt; m += rows) {
        for (uint32_t n = col; n < nt; n += cols) {
            cb_wait_front(16, 1);
            noc_async_write_page(m * nt + n, c, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
}
