"""Export and validate the actual NPU row-sum inputs before CUDA benchmarking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

r = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
out = args.output.resolve()
scratch = out / 'scratch'
scratch.mkdir(parents=True, exist_ok=True)
metal = Path.home() / 'dev/tenstorrent/tt-metal-v0.79.0'
sim = Path.home() / 'dev/tenstorrent/sim/libttsim_wh.so'
revision = subprocess.check_output(['git', '-C', str(metal), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == 'de546d3b146758714d900f11b218c8f9c805f410'
assert hashlib.sha256(sim.read_bytes()).hexdigest() == '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a'
with (out / 'npu-build.log').open('w') as log:
    subprocess.run(['python3', str(r / 'tools/report/build_reduction.py')], stdout=log, stderr=subprocess.STDOUT, check=True)
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1',
           ACCELFLOW_REDUCTION_EXPORT_DIR=str(scratch))
command = [str(r / 'build/reduction/reduction'), str(out / 'npu-fixtures.json'), '1', '1']
with (out / 'npu-fixtures.log').open('w') as log:
    subprocess.run(command, cwd=metal, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
npu = json.loads((out / 'npu-fixtures.json').read_text())
assert npu['passed'] and len(npu['cases']) == 25
fixtures = []
for case in npu['cases']:
    h, w = case['shape_hw']
    key = f"{case['name']}_{h}x{w}"
    binary = scratch / (key + '.bin')
    assert binary.exists()
    fixtures.append(dict(key=key, binary='scratch/' + binary.name, performance=False,
                         npu_row_sums=case['row_sums'], sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))

# Larger CUDA-only performance inputs: integer BF16 values, deterministic and exact.
# These are not claimed to have been executed on TTSim.
for h, w in [(128, 320), (640, 320), (320, 4097), (1024, 8193)]:
    hp, wp = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    key = f'perf_integer_{h}x{w}'
    binary = scratch / (key + '.bin')
    with binary.open('wb') as f:
        f.write(struct.pack('<8sIIII', b'AFRS0001', h, w, hp, wp))
        for row in range(hp):
            values = [float((row * 13 + col * 7) % 17 - 8) if row < h and col < w else 0.0 for col in range(wp)]
            f.write(b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', x))[0] >> 16) for x in values))
    fixtures.append(dict(key=key, binary='scratch/' + binary.name, performance=True,
                         sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
manifest = dict(passed=True, npu_command=command, tt_metal_revision=revision, fixtures=fixtures)
(out / 'fixtures.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'FIXTURES_PASS npu=25 cuda_performance=4 {out}', flush=True)
