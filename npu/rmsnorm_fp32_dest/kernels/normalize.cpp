#include <cstdint>
#include "fp32_unpack_compat.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

static void publish() {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    pack_tile(0, 16);
    cb_push_back(16, 1);
    tile_regs_release();
}

void kernel_main() {
    compute_kernel_hw_startup(0, 3, 16);
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    const uint32_t inverse_width = get_arg_val<uint32_t>(1);
    const uint32_t epsilon = get_arg_val<uint32_t>(2);
    constexpr uint32_t inputs[] = {0, 1, 2};
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        for (uint32_t cb : inputs) cb_wait_front(cb, 1);
        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inverse_width);
        add_unary_tile(0, epsilon);
        rsqrt_tile_init();
        rsqrt_tile(0);
        copy_init(1);
        copy_tile(1, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_init(2);
        copy_tile(2, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        publish();
        for (uint32_t cb : inputs) cb_pop_front(cb, 1);
    }
}
