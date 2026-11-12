#include <cstdint>
#include "api/dataflow/dataflow_api.h"

static uint32_t face_offset(uint32_t row, uint32_t col) {
    return (row / 16 * 2 + col / 16) * 256 + row % 16 * 16 + col % 16;
}

void kernel_main() {
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ag = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    constexpr auto as = TensorAccessorArgs<ag.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(ax, get_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(ag, get_arg_val<uint32_t>(1));
    const auto stats = TensorAccessor(as, get_arg_val<uint32_t>(2));
    const uint32_t ht = get_arg_val<uint32_t>(3), wt = get_arg_val<uint32_t>(4);
    // Raw scratch for one prefix tile. It is never used as an FPU operand.
    cb_reserve_back(3, 1);
    const uint32_t scratch = get_write_ptr(3);
    for (uint32_t row = 0; row < ht; ++row) {
        noc_async_read_page(row, stats, scratch);
        noc_async_read_barrier();
        const auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t col = 0; col < wt; ++col) {
            for (uint32_t cb : {0U,1U,2U}) cb_reserve_back(cb, 1);
            auto* target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
            // Replicate original FP32 words; do not split or recompute them.
            for (uint32_t r = 0; r < 32; ++r)
                for (uint32_t c = 0; c < 32; ++c)
                    target[face_offset(r,c)] = source[face_offset(31,r)];
            noc_async_read_page(row*wt+col, x, get_write_ptr(1));
            noc_async_read_page(col, gamma, get_write_ptr(2));
            noc_async_read_barrier();
            for (uint32_t cb : {0U,1U,2U}) cb_push_back(cb, 1);
        }
    }
}
