#include <cstdint>
#include "api/dataflow/dataflow_api.h"

static uint32_t face_offset(uint32_t row, uint32_t col) {
    return (row / 16 * 2 + col / 16) * 256 + row % 16 * 16 + col % 16;
}

// 부동소수점 산술 없이 양수 FP32의 mantissa 조각을 FP32 비트로 표현합니다.
static uint32_t residual_bits(uint32_t original, uint32_t remainder) {
    if (remainder == 0) return 0;
    const uint32_t exponent = (original >> 23) & 255;
    uint32_t leading = 0;
    for (uint32_t value = remainder; value > 1; value >>= 1) ++leading;
    const int32_t e = int32_t(exponent) - 23 + int32_t(leading);
    if (e <= 0) return remainder << (exponent ? exponent - 1 : 0);
    return (uint32_t(e) << 23) | ((remainder << (23 - leading)) & 0x7fffff);
}

// Expand 32 row statistics into face-layout tiles without FP arithmetic.
static void publish_stats(const volatile tt_l1_ptr uint32_t* words) {
    for (uint32_t cb : {0U, 4U, 5U}) cb_reserve_back(cb, 1);
    auto* hi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
    auto* middle = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
    auto* low = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5));
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t word = words[r];
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t i = face_offset(r, c);
            hi[i] = word & 0xffffe000;
            middle[i] = residual_bits(word, word & 0x1ff8);
            low[i] = residual_bits(word, word & 7);
        }
    }
    for (uint32_t cb : {0U, 4U, 5U}) cb_push_back(cb, 1);
}

void kernel_main() {
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ag = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(ax, get_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(ag, get_arg_val<uint32_t>(1));
    const uint32_t ht = get_arg_val<uint32_t>(2), wt = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4), local_wt = get_arg_val<uint32_t>(5);
    const uint32_t rank = get_arg_val<uint32_t>(6), peers = get_arg_val<uint32_t>(7);
    const uint32_t ready_addr = get_semaphore(get_arg_val<uint32_t>(8));
    const uint32_t valid_addr = get_semaphore(get_arg_val<uint32_t>(9));
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    auto* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    const uint32_t leader_x = get_arg_val<uint32_t>(10), leader_y = get_arg_val<uint32_t>(11);
    // CB28 is allocated identically on all workers before variable-size local caches.
    // It is a raw mailbox: peers*128 bytes for partials, then 128 for the global sum.
    cb_reserve_back(28, 1);
    const uint32_t mailbox = get_write_ptr(28);
    const uint32_t global_slot = mailbox + peers * 128;
    auto* own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox + rank * 128);
    auto* global = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_slot);

    for (uint32_t row = 0; row < ht; ++row) {
        cb_reserve_back(24, local_wt);
        const uint32_t retained = get_write_ptr(24);
        for (uint32_t col = 0; col < local_wt; ++col)
            noc_async_read_page(row * wt + start + col, input, retained + col * 4096);
        noc_async_read_barrier();
        cb_push_back(24, local_wt);

        cb_wait_front(25, local_wt);
        const uint32_t squares = get_read_ptr(25);
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t col = 0; col < local_wt; ++col) {
                cb_reserve_back(0, 1);
                const auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(squares + col * 4096);
                auto* target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
                for (uint32_t i = 0; i < 1024; ++i) {
                    const uint32_t word = source[i];
                    target[i] = pass == 0 ? word & 0xffffe000 : residual_bits(word, word & 0x1fff);
                }
                cb_push_back(0, 1);
            }
        }
        cb_pop_front(25, local_wt);

        // Gather one FP32 partial per row. Completion precedes the ready signal.
        cb_wait_front(26, 1);
        const auto* partial = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(26));
        for (uint32_t r = 0; r < 32; ++r) own[r] = partial[face_offset(31, r)];
        cb_pop_front(26, 1);
        if (rank != 0) {
            // Reset before announcing this row, so the leader cannot race the reset.
            noc_semaphore_set(valid, 0);
            noc_async_write(mailbox + rank * 128,
                get_noc_addr(leader_x, leader_y, mailbox + rank * 128), 128);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(leader_x, leader_y, ready_addr), 1);
            noc_semaphore_wait(valid, 1);
        } else {
            if (peers > 1) {
                noc_semaphore_wait(ready, peers - 1);
                noc_semaphore_set(ready, 0);
            }
            for (uint32_t peer = 0; peer < peers; ++peer)
                publish_stats(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox + peer * 128));
            cb_wait_front(27, 1);
            const auto* total = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(27));
            for (uint32_t r = 0; r < 32; ++r) global[r] = total[face_offset(r, 0)];
            cb_pop_front(27, 1);
            for (uint32_t peer = 1; peer < peers; ++peer) {
                const uint32_t x = get_arg_val<uint32_t>(10 + peer * 2);
                const uint32_t y = get_arg_val<uint32_t>(11 + peer * 2);
                noc_async_write(global_slot, get_noc_addr(x, y, global_slot), 128);
            }
            noc_async_write_barrier();
            for (uint32_t peer = 1; peer < peers; ++peer) {
                const uint32_t x = get_arg_val<uint32_t>(10 + peer * 2);
                const uint32_t y = get_arg_val<uint32_t>(11 + peer * 2);
                noc_semaphore_inc(get_noc_addr(x, y, valid_addr), 1);
            }
        }

        // A worker cannot announce the next row until it has consumed this total.
        // Thus the next gather also acts as credit for reusing the global mailbox.
        for (uint32_t col = 0; col < local_wt; ++col) {
            publish_stats(global);
            cb_reserve_back(1, 1);
            cb_reserve_back(2, 1);
            const auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(retained + col * 4096);
            auto* target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
            for (uint32_t i = 0; i < 1024; ++i) target[i] = source[i];
            noc_async_read_page(start + col, gamma, get_write_ptr(2));
            noc_async_read_barrier();
            cb_push_back(1, 1);
            cb_push_back(2, 1);
        }
    }
}
