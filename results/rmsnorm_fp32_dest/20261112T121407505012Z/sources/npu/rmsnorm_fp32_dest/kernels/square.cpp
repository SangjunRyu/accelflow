#include <cstdint>
#include "fp32_unpack_compat.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"

static void publish() {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    pack_tile(0, 16);
    cb_push_back(16, 1);
    tile_regs_release();
}

// True FP32 input is copied to Dest before SFPU multiplication.
void kernel_main() {
    compute_kernel_hw_startup(0, 3, 16);
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        cb_wait_front(0, 1);
        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, 0, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 0);
        publish();
        cb_pop_front(0, 1);
    }
}
