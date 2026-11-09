"""Validate the official C ABI in an isolated process, without running firmware."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from datetime import datetime, timezone

library, tag, url, expected, output = sys.argv[1:]
path = Path(library).resolve()
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'SHA-256 mismatch'
def capture(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

dynamic = capture('readelf', '-d', str(path))
dependencies = capture('ldd', str(path))
assert 'not found' not in dependencies, 'Unresolved dependency'
elf = capture('readelf', '-h', str(path))
assert 'ELF64' in elf and 'little endian' in elf and 'X86-64' in elf, 'Wrong ELF target'
lib = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
names = ('libttsim_init', 'libttsim_exit', 'libttsim_clock',
         'libttsim_set_pci_dma_mem_callbacks', 'libttsim_pci_config_rd32',
         'libttsim_pci_config_wr32', 'libttsim_pci_mem_rd_bytes',
         'libttsim_pci_mem_wr_bytes', 'libttsim_tile_rd_bytes', 'libttsim_tile_wr_bytes')
for name in names:
    getattr(lib, name)
print('RTLD_NOW load and documented ABI symbol resolution: PASS', flush=True)
for name in ('libttsim_init', 'libttsim_exit'):
    getattr(lib, name).argtypes = []
    getattr(lib, name).restype = None
lib.libttsim_pci_config_rd32.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
lib.libttsim_pci_config_rd32.restype = ctypes.c_uint32
lib.libttsim_pci_config_wr32.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
lib.libttsim_pci_config_wr32.restype = None
lib.libttsim_clock.argtypes = [ctypes.c_uint32]
lib.libttsim_clock.restype = None
# Pinned source: src/libttsim.cpp and src/sim.cpp. DRAM is lazy anonymous mmap;
# no workload, DMA, tile access, or DRAM sweep is performed here.
print('Initializing one virtual Wormhole (no firmware/workload)...', flush=True)
lib.libttsim_init()
try:
    identity = lib.libttsim_pci_config_rd32(0, 0)
    assert identity == 0x401E1E52, f'Unexpected Wormhole PCI identity: {identity:#x}'
    assert lib.libttsim_pci_config_rd32(1, 0) == 0xFFFFFFFF, 'Nonzero function should be absent'
    assert lib.libttsim_pci_config_rd32(8, 0) == 0xFFFFFFFF, 'Second device should be absent'
    bars = {hex(offset): hex(lib.libttsim_pci_config_rd32(0, offset))
            for offset in (0x10, 0x14, 0x18, 0x1c, 0x20, 0x24)}
    # Cache-line-size register is documented as writable; restore immediately.
    original = lib.libttsim_pci_config_rd32(0, 0x0c)
    lib.libttsim_pci_config_wr32(0, 0x0c, (original & ~0xff) | 0x10)
    assert lib.libttsim_pci_config_rd32(0, 0x0c) & 0xff == 0x10
    lib.libttsim_pci_config_wr32(0, 0x0c, original)
    assert lib.libttsim_pci_config_rd32(0, 0x0c) == original
    lib.libttsim_clock(1)
finally:
    lib.libttsim_exit()
print(f'Lifecycle, PCI identity {identity:#010x}, absent-slot reads, config roundtrip, '
      'and one idle clock step: PASS', flush=True)
report = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'release': tag, 'download_url': url, 'library_path': str(path),
    'file_size_bytes': path.stat().st_size, 'sha256': expected,
    'file': capture('file', str(path)).strip(),
    'elf_header': elf, 'dynamic_section': dynamic,
    'dynamic_dependencies': dependencies,
    'symbol_versions': capture('readelf', '--version-info', str(path)),
    'exported_symbols': capture('nm', '-D', '--defined-only', str(path)),
    'rtld_now_load': 'pass', 'required_symbols': list(names),
    'lifecycle': 'pass', 'pci_vendor_device_id': f'{identity:#010x}',
    'pci_bars': bars, 'absent_pci_slots': 'pass', 'config_register_roundtrip': 'pass',
    'idle_clock_steps': 1,
    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'tt_metal_smoke_test': 'not run',
    'limitations': 'Standalone lifecycle/PCI checks do not establish TT-Metal compatibility or kernel correctness.'
}
Path(output).write_text(json.dumps(report, indent=2) + '\n')
