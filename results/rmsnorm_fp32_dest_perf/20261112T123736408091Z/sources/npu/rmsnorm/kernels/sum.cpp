#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/cumsum.h"
#include "api/compute/eltwise_binary_sfpu.h"

// hi/low를 각각 누적한 뒤 SFPU에서 합칩니다. 두 pass 사이에 DST를 유지하고,
// cumsum의 carry를 건드리는 다른 SFPU 연산은 두 scan이 끝난 뒤 수행합니다.
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    const uint32_t ht = get_arg_val<uint32_t>(0), wt = get_arg_val<uint32_t>(1);
    for (uint32_t row = 0; row < ht; ++row) {
        tile_regs_acquire();
        cumsum_tile_init();
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t col = 0; col < wt; ++col) {
                cb_wait_front(0, 1);
                transpose_init(0);
                transpose_tile(0, 0, pass);
                cumsum_tile(pass, col == 0);
                cb_pop_front(0, 1);
            }
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(16, 1);
        pack_tile(0, 16);
        cb_push_back(16, 1);
        tile_regs_release();
    }
}
