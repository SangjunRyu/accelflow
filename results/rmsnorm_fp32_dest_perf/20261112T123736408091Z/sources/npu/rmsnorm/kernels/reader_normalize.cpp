#include <cstdint>
#include "api/dataflow/dataflow_api.h"

static uint32_t face_offset(uint32_t row, uint32_t col) {
    return (row / 16 * 2 + col / 16) * 256 + row % 16 * 16 + col % 16;
}

// 양수 FP32의 작은 mantissa 조각을 다시 FP32 비트로 표현합니다.
// 부동소수점 합산은 하지 않습니다. 세 조각은 각각 TF32 경로에서도 정확히 읽힙니다.
static uint32_t residual_bits(uint32_t original, uint32_t remainder) {
    if (remainder == 0) return 0;
    const uint32_t exponent = (original >> 23) & 255;
    uint32_t leading = 0;
    for (uint32_t v = remainder; v > 1; v >>= 1) ++leading;
    const int32_t new_exponent = int32_t(exponent) - 23 + int32_t(leading);
    if (new_exponent <= 0) return remainder << (exponent ? exponent - 1 : 0);
    return (uint32_t(new_exponent) << 23) | ((remainder << (23 - leading)) & 0x7fffff);
}

void kernel_main() {
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ag = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    constexpr auto as = TensorAccessorArgs<ag.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(ax, get_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(ag, get_arg_val<uint32_t>(1));
    const auto stats = TensorAccessor(as, get_arg_val<uint32_t>(2));
    constexpr uint32_t buffers[] = {0, 1, 2, 4, 5};
    const uint32_t ht = get_arg_val<uint32_t>(3), wt = get_arg_val<uint32_t>(4);

    for (uint32_t row = 0; row < ht; ++row) {
        // CB3는 한 row group의 마지막 prefix 타일을 보관하는 scratch입니다.
        // RISC-V는 마지막 prefix의 비트만 복사하며 합·sqrt를 계산하지 않습니다.
        cb_reserve_back(3, 1);
        const uint32_t scratch = get_write_ptr(3);
        noc_async_read_page(row, stats, scratch);
        noc_async_read_barrier();
        cb_push_back(3, 1);
        cb_wait_front(3, 1);
        const auto* sums = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(3));

        for (uint32_t col = 0; col < wt; ++col) {
            for (uint32_t cb : buffers) cb_reserve_back(cb, 1);
            auto* broadcast = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
            auto* middle = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
            auto* low = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5));
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t word = sums[face_offset(31, r)];
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t i = face_offset(r, c);
                    broadcast[i] = word & 0xffffe000;
                    middle[i] = residual_bits(word, word & 0x1ff8);
                    low[i] = residual_bits(word, word & 7);
                }
            }
            noc_async_read_page(row * wt + col, input, get_write_ptr(1));
            noc_async_read_page(col, gamma, get_write_ptr(2));
            noc_async_read_barrier();
            for (uint32_t cb : buffers) cb_push_back(cb, 1);
        }
        cb_pop_front(3, 1);
    }
}
