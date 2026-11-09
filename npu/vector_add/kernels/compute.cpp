#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

// Tensix의 타일 덧셈: 입력 CB0/CB1 → destination register → 출력 CB16.
void kernel_main() {
    constexpr auto a = tt::CBIndex::c_0;
    constexpr auto b = tt::CBIndex::c_1;
    constexpr auto output = tt::CBIndex::c_16;
    compute_kernel_hw_startup(a, b, output);
    add_init(a, b);
    for (uint32_t tile = 0; tile < get_arg_val<uint32_t>(0); ++tile) {
        // 두 입력이 준비되면 registers를 확보해 타일 하나를 계산합니다.
        cb_wait_front(a, 1);
        cb_wait_front(b, 1);
        tile_regs_acquire();
        add_tiles(a, b, 0, 0, 0);

        // 계산 완료 후 pack하여 출력 CB에 공개하고, 사용한 입력/레지스터를 반환합니다.
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(output, 1);
        pack_tile(0, output);
        cb_push_back(output, 1);
        cb_pop_front(a, 1);
        cb_pop_front(b, 1);
        tile_regs_release();
    }
}
