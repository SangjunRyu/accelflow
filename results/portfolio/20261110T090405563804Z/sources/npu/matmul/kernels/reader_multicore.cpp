#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 각 코어가 담당한 출력 타일 구간에 필요한 A/B를 DRAM에서 직접 읽습니다.
// 다른 코어와 입력 CB를 공유하지 않는 단순 multicore 기준 구현입니다.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t kt = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    for (uint32_t local = 0; local < count; ++local) {
        // 전역 C 타일 번호에서 행/열을 복원합니다. 담당 구간이 행 경계를 넘어도 같습니다.
        const uint32_t output = start + local;
        const uint32_t m = output / nt;
        const uint32_t n = output % nt;
        for (uint32_t k = 0; k < kt; ++k) {
            cb_reserve_back(0, 1);
            noc_async_read_page(m * kt + k, a, get_write_ptr(0));
            noc_async_read_barrier();
            cb_push_back(0, 1);

            cb_reserve_back(1, 1);
            noc_async_read_page(k * nt + n, b, get_write_ptr(1));
            noc_async_read_barrier();
            cb_push_back(1, 1);
        }
    }
}
