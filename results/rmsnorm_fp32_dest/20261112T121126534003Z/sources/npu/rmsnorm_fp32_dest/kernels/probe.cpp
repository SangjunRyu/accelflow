#include <cstdint>
#include "fp32_unpack_compat.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"

// Test only CB -> Dest -> pack. No arithmetic can hide unpack precision loss.
void kernel_main() {
    // Keep the startup SrcB companion in Default mode. The unary LLK emits dummy SrcB unpacks.
    compute_kernel_hw_startup(0, 3, 16);
    const uint32_t operation = get_arg_val<uint32_t>(0);
    cb_wait_front(0, 1);
    tile_regs_acquire();
    if (operation == 0) {
        copy_init(0);
        copy_tile(0, 0, 0);
    } else {
        transpose_init(0);
        transpose_tile(0, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    pack_tile(0, 16);
    cb_push_back(16, 1);
    tile_regs_release();
    cb_pop_front(0, 1);
}
