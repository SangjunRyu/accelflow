#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// A의 행 multicast와 B의 열 multicast는 서로 다른 semaphore 쌍을 사용합니다.
struct Channel {
    uint32_t cb, peers;
    bool sender;
    uint32_t sx, sy, first_x, first_y, last_x, last_y, ready, valid;
};

inline Channel channel(uint32_t arg, uint32_t cb, uint32_t peers, bool sender) {
    return {cb, peers, sender,
            get_arg_val<uint32_t>(arg), get_arg_val<uint32_t>(arg + 1),
            get_arg_val<uint32_t>(arg + 2), get_arg_val<uint32_t>(arg + 3),
            get_arg_val<uint32_t>(arg + 4), get_arg_val<uint32_t>(arg + 5),
            get_semaphore(get_arg_val<uint32_t>(arg + 6)),
            get_semaphore(get_arg_val<uint32_t>(arg + 7))};
}

template <typename Accessor>
inline void share_tile(const Channel& ch, const Accessor& input, uint32_t tile, bool exists) {
    // 유효하지 않은 경계 행/열도 reserve/push하여 모든 코어의 ring 위치를 맞춥니다.
    cb_reserve_back(ch.cb, 1);
    const uint32_t write = get_write_ptr(ch.cb);
    if (exists) {
        auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ch.ready);
        auto* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ch.valid);
        if (ch.sender) {
            noc_async_read_page(tile, input, write);
            noc_async_read_barrier();
            if (ch.peers > 1) {
                noc_semaphore_wait(ready, ch.peers - 1);
                noc_semaphore_set(ready, 0);
                const uint64_t data = get_noc_multicast_addr(
                    ch.first_x, ch.first_y, ch.last_x, ch.last_y, write);
                noc_async_write_multicast(write, data, 2048, ch.peers - 1);
                noc_async_write_barrier();
                const uint64_t flag = get_noc_multicast_addr(
                    ch.first_x, ch.first_y, ch.last_x, ch.last_y, ch.valid);
                noc_semaphore_set_multicast(ch.valid, flag, ch.peers - 1);
                noc_async_write_barrier();
            }
        } else {
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(get_noc_addr(ch.sx, ch.sy, ch.ready), 1);
            noc_semaphore_wait(valid, 1);
        }
    }
    // exists=false인 슬롯은 compute가 값을 읽지 않고 pop하므로 초기화가 필요 없습니다.
    cb_push_back(ch.cb, 1);
}

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t mt = get_arg_val<uint32_t>(2);
    const uint32_t kt = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    const uint32_t row = get_arg_val<uint32_t>(5);
    const uint32_t col = get_arg_val<uint32_t>(6);
    const uint32_t rows = get_arg_val<uint32_t>(7);
    const uint32_t cols = get_arg_val<uint32_t>(8);
    const Channel a_channel = channel(9, 0, cols, col == 0);
    const Channel b_channel = channel(17, 1, rows, row == 0);
    if (a_channel.sender)
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_channel.valid), 1);
    if (b_channel.sender)
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_channel.valid), 1);

    const uint32_t m_groups = (mt + rows - 1) / rows;
    const uint32_t n_groups = (nt + cols - 1) / cols;
    for (uint32_t mg = 0; mg < m_groups; ++mg) {
        const uint32_t m = mg * rows + row;
        for (uint32_t ng = 0; ng < n_groups; ++ng) {
            const uint32_t n = ng * cols + col;
            for (uint32_t k = 0; k < kt; ++k) {
                // 모든 코어가 A→B 순서로 진행하며 행/열 ready 통보 순서를 맞춥니다.
                share_tile(a_channel, a, m * kt + k, m < mt);
                share_tile(b_channel, b, k * nt + n, n < nt);
            }
        }
    }
}
