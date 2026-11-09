#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB0: 행별 sum(x²)를 반복한 타일, CB1: 입력, CB2: gamma.
// 평균·epsilon·rsqrt·정규화·가중치 적용은 compute engine에서 수행합니다.
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    const uint32_t inverse_width_bits = get_arg_val<uint32_t>(1);
    const uint32_t epsilon_bits = get_arg_val<uint32_t>(2);

    constexpr uint32_t buffers[] = {0, 1, 2, 4, 5};
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        for (uint32_t cb : buffers) cb_wait_front(cb, 1);
        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, 0, 0);
        // FP32 통계값은 hi/middle/low로 전달하고 SFPU에서 정확히 재구성합니다.
        copy_init(4);
        copy_tile(4, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        copy_init(5);
        copy_tile(5, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inverse_width_bits);
        add_unary_tile(0, epsilon_bits);
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

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(16, 1);
        pack_tile(0, 16);
        cb_push_back(16, 1);
        tile_regs_release();
        for (uint32_t cb : buffers) cb_pop_front(cb, 1);
    }
}
