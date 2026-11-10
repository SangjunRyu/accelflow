#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

// K chunk 사이에도 네 destination tiles를 유지합니다.
// 중간 결과를 BF16 CB에 pack하지 않아 추가 양자화와 partial-result 전송을 피합니다.
void kernel_main() {
    const uint32_t mt = get_arg_val<uint32_t>(0);
    const uint32_t kt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t chunk = get_arg_val<uint32_t>(3);
    const uint32_t slots = 2 * chunk;
    constexpr auto a = tt::CBIndex::c_0;
    constexpr auto b = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, out);
    matmul_init(a, b);

    for (uint32_t m = 0; m < mt; m += 2) {
        const uint32_t rows = mt - m >= 2 ? 2 : 1;
        for (uint32_t n = 0; n < nt; n += 2) {
            const uint32_t cols = nt - n >= 2 ? 2 : 1;

            // 출력 블록마다 한 번만 acquire하고 K는 항상 오름차순으로 누적합니다.
            tile_regs_acquire();
            for (uint32_t start = 0; start < kt; start += chunk) {
                const uint32_t count = kt - start < chunk ? kt - start : chunk;
                cb_wait_front(a, slots);
                cb_wait_front(b, slots);

                for (uint32_t k = 0; k < count; ++k) {
                    for (uint32_t r = 0; r < rows; ++r) {
                        for (uint32_t c = 0; c < cols; ++c) {
                            matmul_tiles(a, b, r * chunk + k, k * 2 + c, r * 2 + c);
                        }
                    }
                }

                // 입력만 해제합니다. destination 누적값은 다음 chunk까지 유지합니다.
                cb_pop_front(a, slots);
                cb_pop_front(b, slots);
            }

            // K 전체를 완료한 뒤에만 BF16 출력을 pack합니다.
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < cols; ++c) {
                    cb_reserve_back(out, 1);
                    pack_tile(r * 2 + c, out);
                    cb_push_back(out, 1);
                }
            }
            tile_regs_release();
        }
    }
}
