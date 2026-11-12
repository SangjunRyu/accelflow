#include <cstdint>
#include "fp32_unpack_compat.hpp"
#include "api/compute/transpose.h"
#include "api/compute/cumsum.h"

static void publish() {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    pack_tile(0, 16);
    cb_push_back(16, 1);
    tile_regs_release();
}

// One full-precision scan: no hi/low passes or residual reconstruction.
void kernel_main() {
    compute_kernel_hw_startup(0, 3, 16);
    const uint32_t ht = get_arg_val<uint32_t>(0);
    const uint32_t wt = get_arg_val<uint32_t>(1);
    for (uint32_t row = 0; row < ht; ++row) {
        tile_regs_acquire();
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(0, 1);
            transpose_init(0);
            transpose_tile(0, 0, 0);
            // Restore the SFPU program/counters after the Dest transpose.
            // The init does not clear cumsum's LREG carry; first=true does.
            cumsum_tile_init();
            cumsum_tile(0, col == 0);
            cb_pop_front(0, 1);
        }
        publish();
    }
}
