#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The reader moves unmodified FP32 words. No mantissa splitting or float math.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t count = get_arg_val<uint32_t>(1);
    for (uint32_t tile = 0; tile < count; ++tile) {
        cb_reserve_back(0, 1);
        noc_async_read_page(tile, input, get_write_ptr(0));
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
}
