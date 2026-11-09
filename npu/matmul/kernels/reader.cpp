// API sequence adapted from Tenstorrent's Apache-2.0 v0.79.0 single-core MatMul example.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// DRAM → L1 입력 CB. 생산자는 reserve → DMA 완료 확인 → push 순서를 지킵니다.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t mt = get_arg_val<uint32_t>(2), kt = get_arg_val<uint32_t>(3),
                   nt = get_arg_val<uint32_t>(4);
    const bool reuse = get_arg_val<uint32_t>(5) != 0;
    for (uint32_t m = 0; m < mt; ++m) {
        // A(m,:) 전체를 한 번만 읽습니다. 다음 행의 reserve는 compute의 pop을 기다립니다.
        if (reuse) {
            // Retain this entire A tile row until all N output columns consume it.
            cb_reserve_back(0, kt);
            const uint32_t write = get_write_ptr(0);
            for (uint32_t k = 0; k < kt; ++k)
                noc_async_read_page(m * kt + k, a, write + k * 2048);
            noc_async_read_barrier();
            cb_push_back(0, kt);
        }

        // 출력 C(m,n)마다 K축 입력 쌍을 전달합니다. A/B tile index는 row-major입니다.
        for (uint32_t n = 0; n < nt; ++n) {
            for (uint32_t k = 0; k < kt; ++k) {
                // 기본 모드는 출력 열마다 A(m,k)를 다시 읽습니다.
                if (!reuse) {
                    cb_reserve_back(0, 1);
                    noc_async_read_page(m * kt + k, a, get_write_ptr(0));
                    noc_async_read_barrier();
                    cb_push_back(0, 1);
                }

                // B(k,n)는 두 모드 모두 streaming합니다. barrier 이후에만 compute에 공개합니다.
                cb_reserve_back(1, 1);
                noc_async_read_page(k * nt + n, b, get_write_ptr(1));
                noc_async_read_barrier();
                cb_push_back(1, 1);
            }
        }
    }
}
