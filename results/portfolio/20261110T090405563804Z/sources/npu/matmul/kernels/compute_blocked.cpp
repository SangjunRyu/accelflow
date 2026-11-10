#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

// A/B의 K 전체를 보관하고 최대 네 출력 타일을 destination registers에 동시 누적합니다.
// API 초기화와 matmul_tiles는 고정된 upstream v0.79.0의 단일 코어 예제를 따릅니다.
void kernel_main() {
    const uint32_t mt = get_arg_val<uint32_t>(0);
    const uint32_t kt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    constexpr auto a = tt::CBIndex::c_0;
    constexpr auto b = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;
    const uint32_t block_tiles = 2 * kt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, out);
    matmul_init(a, b);

    for (uint32_t m = 0; m < mt; m += 2) {
        const uint32_t rows = mt - m >= 2 ? 2 : 1;
        for (uint32_t n = 0; n < nt; n += 2) {
            const uint32_t cols = nt - n >= 2 ? 2 : 1;
            cb_wait_front(a, block_tiles);
            cb_wait_front(b, block_tiles);

            // 출력 register index = r*2+c. 각 출력의 K 누적 순서는 기존 버전과 같습니다.
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; ++k) {
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        matmul_tiles(a, b, r * kt + k, k * 2 + c, r * 2 + c);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();

            // 유효한 출력만 블록 내부 row-major 순서로 writer에 전달합니다.
            // output CB가 depth 1이어도 한 타일씩 넘기므로 네 타일 reserve가 필요 없습니다.
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < cols; ++c) {
                    cb_reserve_back(out, 1);
                    pack_tile(r * 2 + c, out);
                    cb_push_back(out, 1);
                }
            }
            tile_regs_release();

            // 입력 두 행/두 열의 모든 재사용이 끝난 뒤에만 reader에 공간을 반환합니다.
            cb_pop_front(a, block_tiles);
            cb_pop_front(b, block_tiles);
        }
    }
}
