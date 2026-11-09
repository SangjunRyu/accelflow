// API sequence adapted from Tenstorrent's Apache-2.0 v0.79.0 single-core MatMul example.
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

// L1 입력 CB → Tensix matmul → 출력 CB. compile arg로 A 재사용 모드를 선택합니다.
void kernel_main() {
    constexpr bool reuse = get_compile_time_arg_val(0) != 0;
    const uint32_t mt = get_arg_val<uint32_t>(0), kt = get_arg_val<uint32_t>(1),
                   nt = get_arg_val<uint32_t>(2);
    constexpr auto a = tt::CBIndex::c_0, b = tt::CBIndex::c_1, out = tt::CBIndex::c_16;

    // 공식 단일 코어 MatMul 예제의 초기화 순서와 source 순서를 따릅니다.
    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, out);
    matmul_init(a, b);
    for (uint32_t m = 0; m < mt; ++m) {
        // 재사용 시 A 한 타일 행 전체가 준비될 때까지 기다리고 모든 n 계산 동안 보관합니다.
        if constexpr (reuse)
            cb_wait_front(a, kt);
        for (uint32_t n = 0; n < nt; ++n) {
            // 출력 타일 하나의 destination registers를 확보하고 Kt개의 부분 곱을 누적합니다.
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; ++k) {
                if constexpr (!reuse)
                    cb_wait_front(a, 1);
                cb_wait_front(b, 1);

                // 재사용 모드는 보관된 A의 k번째 타일, 기본 모드는 CB 앞의 타일을 사용합니다.
                matmul_tiles(a, b, reuse ? k : 0, 0, 0);
                if constexpr (!reuse)
                    cb_pop_front(a, 1);
                cb_pop_front(b, 1);
            }

            // 연산 완료를 확인하고 BF16 출력으로 pack한 뒤 writer에 넘깁니다.
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(out, 1);
            pack_tile(0, out);
            cb_push_back(out, 1);
            tile_regs_release();
        }

        // 모든 출력 열이 이 A 행을 사용한 뒤에만 해제합니다.
        if constexpr (reuse)
            cb_pop_front(a, kt);
    }
}
