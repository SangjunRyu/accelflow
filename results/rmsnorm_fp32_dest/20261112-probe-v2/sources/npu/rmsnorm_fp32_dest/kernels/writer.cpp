#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t count = get_arg_val<uint32_t>(1);
    for (uint32_t tile = 0; tile < count; ++tile) {
        cb_wait_front(16, 1);
        noc_async_write_page(tile, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
