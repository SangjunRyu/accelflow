#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 2×2 출력 블록의 입력을 K 방향 chunk 타일씩 공급합니다.
// CB 용량은 전체 Kt가 아닌 chunk 크기에 비례합니다.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t mt = get_arg_val<uint32_t>(2);
    const uint32_t kt = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    const uint32_t chunk = get_arg_val<uint32_t>(6);
    const uint32_t slots = 2 * chunk;

    for (uint32_t m = 0; m < mt; m += 2) {
        const uint32_t rows = mt - m >= 2 ? 2 : 1;
        for (uint32_t n = 0; n < nt; n += 2) {
            const uint32_t cols = nt - n >= 2 ? 2 : 1;
            for (uint32_t start = 0; start < kt; start += chunk) {
                const uint32_t count = kt - start < chunk ? kt - start : chunk;

                // 마지막 K chunk나 홀수 출력 경계도 고정 슬롯을 reserve합니다.
                // 유효한 입력만 읽고 빈 슬롯은 compute가 접근하지 않습니다.
                cb_reserve_back(0, slots);
                cb_reserve_back(1, slots);
                const uint32_t a_write = get_write_ptr(0);
                const uint32_t b_write = get_write_ptr(1);

                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t k = 0; k < count; ++k) {
                        const uint32_t tile = (m + r) * kt + start + k;
                        noc_async_read_page(tile, a, a_write + (r * chunk + k) * 2048);
                    }
                }
                for (uint32_t k = 0; k < count; ++k) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        const uint32_t tile = (start + k) * nt + n + c;
                        noc_async_read_page(tile, b, b_write + (k * 2 + c) * 2048);
                    }
                }

                // DMA 완료 후 공개합니다. compute의 pop이 다음 chunk 공간을 확보합니다.
                noc_async_read_barrier();
                cb_push_back(0, slots);
                cb_push_back(1, slots);
            }
        }
    }
}
