#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// scaler는 전체 실행 동안 한 타일을 재사용합니다. 입력은 타일마다 streaming합니다.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto scale_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto scale = TensorAccessor(scale_args, get_arg_val<uint32_t>(1));
    const uint32_t tiles = get_arg_val<uint32_t>(2);

    cb_reserve_back(1, 1);
    noc_async_read_page(0, scale, get_write_ptr(1));
    noc_async_read_barrier();
    cb_push_back(1, 1);

    for (uint32_t tile = 0; tile < tiles; ++tile) {
        cb_reserve_back(0, 1);
        noc_async_read_page(tile, x, get_write_ptr(0));
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
}
