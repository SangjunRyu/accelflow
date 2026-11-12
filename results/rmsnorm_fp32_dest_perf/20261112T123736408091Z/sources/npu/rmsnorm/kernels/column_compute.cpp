#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/cumsum.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

// 한 번 초기화한 FP32 packer로 같은 형식의 서로 다른 CB에 기록합니다.
static void publish(uint32_t output) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(output, 1);
    pack_tile(0, output);
    cb_push_back(output, 1);
    tile_regs_release();
}

// CB24: 이 row group의 원본 입력. 정규화가 끝날 때까지 pop하지 않습니다.
// CB25: 제곱 타일 캐시. DRAM을 거치지 않고 reader가 두 조각으로 변환합니다.
static void square_row(uint32_t width_tiles) {
    cb_wait_front(24, width_tiles);
    for (uint32_t col = 0; col < width_tiles; ++col) {
        tile_regs_acquire();
        copy_init(24);
        copy_tile(24, col, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 0);
        publish(25);
    }
}

// 두 scan의 carry는 다른 SFPU 연산과 섞지 않습니다.
// CB26은 마지막 prefix 하나이며 reader가 행별 합을 추출합니다.
static void sum_row(uint32_t width_tiles) {
    tile_regs_acquire();
    cumsum_tile_init();
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t col = 0; col < width_tiles; ++col) {
            cb_wait_front(0, 1);
            transpose_init(0);
            transpose_tile(0, 0, pass);
            cumsum_tile(pass, col == 0);
            cb_pop_front(0, 1);
        }
    }
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    publish(26);
}

static void normalize_row(uint32_t width_tiles, uint32_t inverse_width, uint32_t epsilon) {
    constexpr uint32_t inputs[] = {0, 1, 2, 4, 5};
    for (uint32_t col = 0; col < width_tiles; ++col) {
        for (uint32_t cb : inputs) cb_wait_front(cb, 1);
        tile_regs_acquire();

        // 통계값의 hi/middle/low를 FP32 DST에서 재구성합니다.
        copy_init(0);
        copy_tile(0, 0, 0);
        copy_init(4);
        copy_tile(4, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        copy_init(5);
        copy_tile(5, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
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
        publish(16);
        for (uint32_t cb : inputs) cb_pop_front(cb, 1);
    }
}

// Leader combines exact FP32 fragments of one partial sum per column shard.
// All arithmetic remains on the compute engine; the reader only rearranges bits.
static void aggregate_partials(uint32_t peers) {
    tile_regs_acquire();
    for (uint32_t peer = 0; peer < peers; ++peer) {
        for (uint32_t cb : {0U, 4U, 5U}) cb_wait_front(cb, 1);
        const uint32_t dest = peer == 0 ? 0 : 1;
        const uint32_t tmp = peer == 0 ? 1 : 2;
        copy_init(0); copy_tile(0, 0, dest);
        copy_init(4); copy_tile(4, 0, tmp);
        add_binary_tile_init(); add_binary_tile(dest, tmp, dest);
        copy_init(5); copy_tile(5, 0, tmp);
        add_binary_tile_init(); add_binary_tile(dest, tmp, dest);
        if (peer != 0) {
            add_binary_tile_init(); add_binary_tile(0, 1, 0);
        }
        for (uint32_t cb : {0U, 4U, 5U}) cb_pop_front(cb, 1);
    }
    publish(27);
}

void kernel_main() {
    compute_kernel_hw_startup(24, 16);
    const uint32_t height_tiles = get_arg_val<uint32_t>(0);
    const uint32_t width_tiles = get_arg_val<uint32_t>(1);
    const uint32_t inverse_width = get_arg_val<uint32_t>(2);
    const uint32_t epsilon = get_arg_val<uint32_t>(3);

    const uint32_t rank = get_arg_val<uint32_t>(4);
    const uint32_t peers = get_arg_val<uint32_t>(5);

    for (uint32_t row = 0; row < height_tiles; ++row) {
        square_row(width_tiles);
        sum_row(width_tiles);
        if (rank == 0) aggregate_partials(peers);
        normalize_row(width_tiles, inverse_width, epsilon);
        cb_pop_front(24, width_tiles);
    }
}
