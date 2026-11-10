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

void kernel_main() {
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ag = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(ax, get_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(ag, get_arg_val<uint32_t>(1));
    const uint32_t ht = get_arg_val<uint32_t>(2), wt = get_arg_val<uint32_t>(3);
    constexpr uint32_t normalize_inputs[] = {0, 1, 2, 4, 5};

    for (uint32_t row = 0; row < ht; ++row) {
        // CB24의 전체 용량이 정확히 한 row group이므로 매번 base로 wrap합니다.
        cb_reserve_back(24, wt);
        const uint32_t retained = get_write_ptr(24);
        for (uint32_t col = 0; col < wt; ++col) {
            noc_async_read_page(row * wt + col, input, retained + col * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(24, wt);

        // compute의 제곱 결과를 L1에서 읽어 hi/low 두 scan으로 전달합니다.
        cb_wait_front(25, wt);
        const uint32_t squares = get_read_ptr(25);
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t col = 0; col < wt; ++col) {
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
        cb_pop_front(25, wt);

        cb_wait_front(26, 1);
        const auto* sums = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(26));
        for (uint32_t col = 0; col < wt; ++col) {
            for (uint32_t cb : normalize_inputs) cb_reserve_back(cb, 1);
            auto* hi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
            auto* middle = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
            auto* low = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5));
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t word = sums[face_offset(31, r)];
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t i = face_offset(r, c);
                    hi[i] = word & 0xffffe000;
                    middle[i] = residual_bits(word, word & 0x1ff8);
                    low[i] = residual_bits(word, word & 7);
                }
            }

            // 원본 입력은 DRAM에서 다시 읽지 않고 CB24의 보관본을 복사합니다.
            const auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(retained + col * 4096);
            auto* target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
            for (uint32_t i = 0; i < 1024; ++i) target[i] = source[i];
            noc_async_read_page(col, gamma, get_write_ptr(2));
            noc_async_read_barrier();
            for (uint32_t cb : normalize_inputs) cb_push_back(cb, 1);
        }
        cb_pop_front(26, 1);
    }
}
