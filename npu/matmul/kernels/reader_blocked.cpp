#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// 출력 2×2 타일 블록에 필요한 A 두 행/B 두 열을 L1에 보관합니다.
// K 전체를 보관하는 첫 blocked 구현이며 Kt <= 4로 host에서 범위를 제한합니다.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t mt = get_arg_val<uint32_t>(2);
    const uint32_t kt = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    const uint32_t block_tiles = 2 * kt;

    for (uint32_t m = 0; m < mt; m += 2) {
        const uint32_t rows = mt - m >= 2 ? 2 : 1;
        for (uint32_t n = 0; n < nt; n += 2) {
            const uint32_t cols = nt - n >= 2 ? 2 : 1;

            // 고정 크기로 reserve하여 CB wrap 때에도 블록이 연속 공간을 차지합니다.
            // 경계에서 없는 행/열의 슬롯은 읽지도 계산하지도 않습니다.
            cb_reserve_back(0, block_tiles);
            cb_reserve_back(1, block_tiles);
            const uint32_t a_write = get_write_ptr(0);
            const uint32_t b_write = get_write_ptr(1);

            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t k = 0; k < kt; ++k) {
                    noc_async_read_page((m + r) * kt + k, a, a_write + (r * kt + k) * 2048);
                }
            }
            for (uint32_t k = 0; k < kt; ++k) {
                for (uint32_t c = 0; c < cols; ++c) {
                    noc_async_read_page(k * nt + n + c, b, b_write + (k * 2 + c) * 2048);
                }
            }

            // 모든 입력 DMA가 끝난 뒤 compute에 블록을 공개합니다.
            noc_async_read_barrier();
            cb_push_back(0, block_tiles);
            cb_push_back(1, block_tiles);
        }
    }
}
