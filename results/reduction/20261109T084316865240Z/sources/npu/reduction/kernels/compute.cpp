// Reduce API sequence checked against TT-Metal de546d3 and its layernorm test kernel.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"

// 열 타일의 행별 부분합을 destination 0에 계속 누적합니다.
void kernel_main() {
    constexpr auto input = tt::CBIndex::c_0;
    constexpr auto scaler = tt::CBIndex::c_1;
    constexpr auto output = tt::CBIndex::c_16;
    const uint32_t row_tiles = get_arg_val<uint32_t>(0);
    const uint32_t col_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(input, scaler, output);
    // REDUCE_ROW SUM의 MVMUL은 scaler→SrcA, input→SrcB 순서로 동작합니다.
    reconfig_data_format(scaler, input);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(input, scaler, output);
    cb_wait_front(scaler, 1);

    for (uint32_t row = 0; row < row_tiles; ++row) {
        tile_regs_acquire();
        for (uint32_t col = 0; col < col_tiles; ++col) {
            cb_wait_front(input, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(input, scaler, 0, 0, 0);
            cb_pop_front(input, 1);
        }

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(output, 1);
        // 결과는 각 행의 column 0에, 나머지 열은 pack mask로 0에 놓입니다.
        pack_tile(0, output);
        cb_push_back(output, 1);
        tile_regs_release();
    }

    cb_pop_front(scaler, 1);
    reduce_uninit(input);
}
