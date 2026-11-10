#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 코어마다 서로 겹치지 않는 전역 C 타일 주소에 기록합니다.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    for (uint32_t local = 0; local < count; ++local) {
        cb_wait_front(16, 1);
        noc_async_write_page(start + local, c, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
