// API and ordering follow pinned TT-Metal's add_2_integers_in_riscv example.
#include "ckernel.h"

// Tensix 없이 RISC-V에서 uint32 덧셈을 수행하는 원소별 기준 구현입니다.
void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_l1 = get_arg_val<uint32_t>(3);
    const uint32_t b_l1 = get_arg_val<uint32_t>(4);
    const uint32_t c_l1 = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);
    const InterleavedAddrGen<true> a{.bank_base_address = a_addr, .page_size = 4};
    const InterleavedAddrGen<true> b{.bank_base_address = b_addr, .page_size = 4};
    const InterleavedAddrGen<true> c{.bank_base_address = c_addr, .page_size = 4};

    // 각 원소의 DRAM 입력을 L1으로 읽고 barrier 이후에 접근합니다.
    for (uint32_t i = 0; i < count; ++i) {
        noc_async_read(a.get_noc_addr(i), a_l1, 4);
        noc_async_read(b.get_noc_addr(i), b_l1, 4);
        noc_async_read_barrier();

        // uint32 덧셈은 modulo 2^32입니다. 아래 visibility 확인 뒤에 NoC 쓰기를 시작합니다.
        *reinterpret_cast<uint32_t*>(c_l1) =
            *reinterpret_cast<uint32_t*>(a_l1) + *reinterpret_cast<uint32_t*>(b_l1);
        // RISC-V stores and NoC reads are independent L1 clients.
        (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c_l1));
        noc_async_write(c_l1, c.get_noc_addr(i), 4);
        noc_async_write_barrier();
    }
}
