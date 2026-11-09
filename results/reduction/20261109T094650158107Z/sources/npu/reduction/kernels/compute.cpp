// Sequence follows the pinned TT-Metal cumsum test's row-wise transpose + scan path.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/cumsum.h"

// 행을 column으로 전치하고 FP32 SFPU prefix sum의 마지막 값을 행 합으로 사용합니다.
void kernel_main() {
    constexpr auto input = tt::CBIndex::c_0;
    constexpr auto output = tt::CBIndex::c_16;
    const uint32_t row_tiles = get_arg_val<uint32_t>(0);
    const uint32_t col_tiles = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(input, output);
    transpose_init(input);
    cumsum_tile_init();

    for (uint32_t row = 0; row < row_tiles; ++row) {
        for (uint32_t col = 0; col < col_tiles; ++col) {
            cb_wait_front(input, 1);
            tile_regs_acquire();
            transpose_init(input);
            transpose_tile(input, 0, 0);
            // 첫 열 타일에서 carry를 초기화하고, 나머지는 이전 타일의 합을 이어받습니다.
            cumsum_tile(0, col == 0);
            tile_regs_commit();
            tile_regs_wait();
            if (col + 1 == col_tiles) {
                cb_reserve_back(output, 1);
                pack_tile(0, output);
                cb_push_back(output, 1);
            }
            tile_regs_release();
            cb_pop_front(input, 1);
        }
    }
}
