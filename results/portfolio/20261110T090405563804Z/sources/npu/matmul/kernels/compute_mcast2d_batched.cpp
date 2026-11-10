#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

// 타일 좌표는 (mg*rows+row, ng*cols+col)이며 경계에서도 A/B CB를 모두 소비합니다.
void kernel_main() {
    const uint32_t mt = get_arg_val<uint32_t>(0);
    const uint32_t kt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t row = get_arg_val<uint32_t>(3);
    const uint32_t col = get_arg_val<uint32_t>(4);
    const uint32_t rows = get_arg_val<uint32_t>(5);
    const uint32_t cols = get_arg_val<uint32_t>(6);
    const uint32_t batch = get_arg_val<uint32_t>(7);
    constexpr auto a = tt::CBIndex::c_0;
    constexpr auto b = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, out);
    matmul_init(a, b);

    for (uint32_t mg = 0; mg < (mt + rows - 1) / rows; ++mg) {
        for (uint32_t ng = 0; ng < (nt + cols - 1) / cols; ++ng) {
            const bool enabled = mg * rows + row < mt && ng * cols + col < nt;
            if (enabled)
                tile_regs_acquire();
            for (uint32_t k = 0; k < kt; k += batch) {
                cb_wait_front(a, batch);
                cb_wait_front(b, batch);
                const uint32_t count = kt-k < batch ? kt-k : batch;
                // K 순서와 FP32 destination 누적은 원래 구현과 같습니다.
                // 마지막 배치의 미사용 슬롯과 경계 코어 슬롯은 계산하지 않습니다.
                if (enabled)
                    for (uint32_t i = 0; i < count; ++i)
                        matmul_tiles(a, b, i, i, 0);
                cb_pop_front(a, batch);
                cb_pop_front(b, batch);
            }
            if (enabled) {
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(out, 1);
                pack_tile(0, out);
                cb_push_back(out, 1);
                tile_regs_release();
            }
        }
    }
}
