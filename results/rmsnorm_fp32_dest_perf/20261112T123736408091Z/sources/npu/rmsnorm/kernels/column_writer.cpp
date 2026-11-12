#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Each row group has a disjoint output interval for this column shard.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t ht = get_arg_val<uint32_t>(1), wt = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    for (uint32_t row = 0; row < ht; ++row) {
        for (uint32_t col = 0; col < count; ++col) {
            cb_wait_front(16, 1);
            noc_async_write_page(row * wt + start + col, output, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
}
