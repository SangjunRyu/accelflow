#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

// 코어별 독립 누적: 출력 타일 하나의 K 전체를 이 코어에서 완료합니다.
// K축 분할/코어 간 partial sum 교환은 하지 않습니다.
void kernel_main() {
    const uint32_t kt = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr auto a = tt::CBIndex::c_0;
    constexpr auto b = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, out);
    matmul_init(a, b);

    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            cb_wait_front(a, 1);
            cb_wait_front(b, 1);
            matmul_tiles(a, b, 0, 0, 0);
            cb_pop_front(a, 1);
            cb_pop_front(b, 1);
        }

        // 같은 코어의 writer가 결과를 소비합니다.
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(out, 1);
        pack_tile(0, out);
        cb_push_back(out, 1);
        tile_regs_release();
    }
}
