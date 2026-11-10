#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 코어별로 배정된 출력 타일 구간만 씁니다. 다른 코어와 주소가 겹치지 않습니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_start = get_arg_val<uint32_t>(2);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        cb_wait_front(16, 1);
        noc_async_write_page(tile_start + tile, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
