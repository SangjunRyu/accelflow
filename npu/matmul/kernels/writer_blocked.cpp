#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Compute의 블록 순서를 DRAM의 전역 tile row-major 순서로 변환합니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t mt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);

    for (uint32_t m = 0; m < mt; m += 2) {
        const uint32_t rows = mt - m >= 2 ? 2 : 1;
        for (uint32_t n = 0; n < nt; n += 2) {
            const uint32_t cols = nt - n >= 2 ? 2 : 1;
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t col = 0; col < cols; ++col) {
                    cb_wait_front(16, 1);
                    const uint32_t tile = (m + r) * nt + n + col;
                    noc_async_write_page(tile, c, get_read_ptr(16));
                    noc_async_write_barrier();
                    cb_pop_front(16, 1);
                }
            }
        }
    }
}
