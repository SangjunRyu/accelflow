#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// NoC0의 한 행에서 rank0은 A sender, 나머지는 receiver입니다.
// 공식 v0.79.0 예제의 reserve → ready → data → valid → push 순서를 따릅니다.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t mt = get_arg_val<uint32_t>(2);
    const uint32_t kt = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    const uint32_t rank = get_arg_val<uint32_t>(5);
    const uint32_t peers = get_arg_val<uint32_t>(6);
    const uint32_t sender_x = get_arg_val<uint32_t>(7);
    const uint32_t sender_y = get_arg_val<uint32_t>(8);
    const uint32_t first_x = get_arg_val<uint32_t>(9);
    const uint32_t first_y = get_arg_val<uint32_t>(10);
    const uint32_t last_x = get_arg_val<uint32_t>(11);
    const uint32_t last_y = get_arg_val<uint32_t>(12);
    const uint32_t ready_addr = get_semaphore(get_arg_val<uint32_t>(13));
    const uint32_t valid_addr = get_semaphore(get_arg_val<uint32_t>(14));
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    auto* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    const uint32_t rounds = (nt + peers - 1) / peers;

    if (rank == 0)
        noc_semaphore_set(valid, 1);

    for (uint32_t m = 0; m < mt; ++m) {
        for (uint32_t group = 0; group < rounds; ++group) {
            const uint32_t n = group * peers + rank;
            for (uint32_t k = 0; k < kt; ++k) {
                // 전 코어가 같은 횟수로 A CB를 진행해 multicast L1 offset을 맞춥니다.
                cb_reserve_back(0, 1);
                const uint32_t write = get_write_ptr(0);
                if (rank == 0) {
                    noc_async_read_page(m * kt + k, a, write);
                    noc_async_read_barrier();

                    if (peers > 1) {
                        // 모든 receiver가 빈 슬롯과 valid=0을 준비했는지 확인합니다.
                        noc_semaphore_wait(ready, peers - 1);
                        noc_semaphore_set(ready, 0);
                        const uint64_t data = get_noc_multicast_addr(
                            first_x, first_y, last_x, last_y, write);
                        noc_async_write_multicast(write, data, 2048, peers - 1);

                        // 학습용 기준 버전은 data 완료 후 valid를 보내도록 명시적으로 기다립니다.
                        noc_async_write_barrier();
                        const uint64_t flag = get_noc_multicast_addr(
                            first_x, first_y, last_x, last_y, valid_addr);
                        noc_semaphore_set_multicast(valid_addr, flag, peers - 1);
                        noc_async_write_barrier();
                    }
                } else {
                    noc_semaphore_set(valid, 0);
                    noc_semaphore_inc(get_noc_addr(sender_x, sender_y, ready_addr), 1);
                    noc_semaphore_wait(valid, 1);
                }
                cb_push_back(0, 1);

                // 존재하는 출력 열에 대해서만 B를 읽습니다. A는 tail receiver에도 전달합니다.
                if (n < nt) {
                    cb_reserve_back(1, 1);
                    noc_async_read_page(k * nt + n, b, get_write_ptr(1));
                    noc_async_read_barrier();
                    cb_push_back(1, 1);
                }
            }
        }
    }
}
