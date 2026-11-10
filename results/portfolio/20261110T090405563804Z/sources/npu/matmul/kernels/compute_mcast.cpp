#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

// 모든 round에서 A를 소비합니다. 출력 없는 tail도 A를 pop해 CB 위치를 맞춥니다.
void kernel_main() {
    const uint32_t mt = get_arg_val<uint32_t>(0);
    const uint32_t kt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t rank = get_arg_val<uint32_t>(3);
    const uint32_t peers = get_arg_val<uint32_t>(4);
    const uint32_t rounds = (nt + peers - 1) / peers;
    constexpr auto a = tt::CBIndex::c_0;
    constexpr auto b = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, out);
    matmul_init(a, b);
    for (uint32_t m = 0; m < mt; ++m) {
        for (uint32_t group = 0; group < rounds; ++group) {
            const bool enabled = group * peers + rank < nt;
            if (enabled)
                tile_regs_acquire();
            for (uint32_t k = 0; k < kt; ++k) {
                cb_wait_front(a, 1);
                if (enabled) {
                    cb_wait_front(b, 1);
                    matmul_tiles(a, b, 0, 0, 0);
                    cb_pop_front(b, 1);
                }
                cb_pop_front(a, 1);
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
