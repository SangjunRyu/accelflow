# Compilation inspection artifacts

Host LLVM IR describes the x86_64 host program; it is not TTIR/TTNN or Tensix IR.
Device files are real SFPI-produced ELF/cache artifacts; cached hash variants may use different configs.

[Manifest](manifest.json) · [Host compile commands](host/compile_commands.json) · [Host LLVM IR](host/matmul-host.ll)

| Kernel | Cache hash | Processor | Bytes | Disassembly | ELF metadata |
| --- | --- | --- | --- | --- | --- |
| compute_streamed | 15131064063784111138 | trisc0 | 263640 | [assembly](device/9059419178545743253/compute_streamed/15131064063784111138/trisc0/assembly.txt) | [ELF](device/9059419178545743253/compute_streamed/15131064063784111138/trisc0/elf-info.txt) |
| compute_streamed | 15131064063784111138 | trisc1 | 260680 | [assembly](device/9059419178545743253/compute_streamed/15131064063784111138/trisc1/assembly.txt) | [ELF](device/9059419178545743253/compute_streamed/15131064063784111138/trisc1/elf-info.txt) |
| compute_streamed | 15131064063784111138 | trisc2 | 230544 | [assembly](device/9059419178545743253/compute_streamed/15131064063784111138/trisc2/assembly.txt) | [ELF](device/9059419178545743253/compute_streamed/15131064063784111138/trisc2/elf-info.txt) |
| reader_streamed | 7468934367456772562 | ncrisc | 263700 | [assembly](device/9059419178545743253/reader_streamed/7468934367456772562/ncrisc/assembly.txt) | [ELF](device/9059419178545743253/reader_streamed/7468934367456772562/ncrisc/elf-info.txt) |
| writer_blocked | 12925800970514938734 | brisc | 206196 | [assembly](device/9059419178545743253/writer_blocked/12925800970514938734/brisc/assembly.txt) | [ELF](device/9059419178545743253/writer_blocked/12925800970514938734/brisc/elf-info.txt) |
