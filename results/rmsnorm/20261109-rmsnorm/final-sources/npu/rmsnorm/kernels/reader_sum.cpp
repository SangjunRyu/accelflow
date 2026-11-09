#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// BF16 값의 제곱은 최대 16개 유효 비트입니다. TF32의 상위 11비트와
// 나머지 조각으로 나누면 두 조각 모두 입력 경로에서 정확히 유지됩니다.
static uint32_t residual_bits(uint32_t word) {
    const uint32_t r = word & 0x1fff;
    if (!r) return 0;
    const uint32_t exponent = (word >> 23) & 255;
    uint32_t leading = 0;
    for (uint32_t v = r; v > 1; v >>= 1) ++leading;
    const int32_t e = int32_t(exponent) - 23 + int32_t(leading);
    if (e <= 0) return r << (exponent ? exponent - 1 : 0);
    return (uint32_t(e) << 23) | ((r << (23 - leading)) & 0x7fffff);
}

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t ht = get_arg_val<uint32_t>(1), wt = get_arg_val<uint32_t>(2);
    for (uint32_t row = 0; row < ht; ++row) {
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t col = 0; col < wt; ++col) {
                cb_reserve_back(0, 1);
                const uint32_t address = get_write_ptr(0);
                noc_async_read_page(row * wt + col, input, address);
                noc_async_read_barrier();
                auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
                for (uint32_t i = 0; i < 1024; ++i) {
                    const uint32_t word = words[i];
                    words[i] = pass == 0 ? word & 0xffffe000 : residual_bits(word);
                }
                cb_push_back(0, 1);
            }
        }
    }
}
