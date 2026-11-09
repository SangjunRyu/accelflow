"""Measure simulator/host dispatch cost; never interpret it as physical NPU latency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--samples', type=int, default=7)
args = parser.parse_args()
if not 5 <= args.samples <= 25:
    raise SystemExit('Samples must be 5..25')

metal = Path(os.environ.get('TT_METAL_HOME', str(Path.home() / 'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim = Path(os.environ.get('TT_METAL_SIMULATOR', str(Path.home() / 'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
revision = subprocess.check_output(['git', '-C', str(metal), 'rev-parse', 'HEAD'], text=True).strip()
if revision != 'de546d3b146758714d900f11b218c8f9c805f410':
    raise SystemExit('Expected pinned TT-Metal v0.79.0')
sim_hash = hashlib.sha256(sim.read_bytes()).hexdigest()
if sim_hash != '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TTSim v1.11.1 Wormhole')
(sim.parent / 'soc_descriptor.yaml').write_bytes((metal / 'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1')
out = root / 'results/matmul_perf' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)

def host_state():
    return {'meminfo': Path('/proc/meminfo').read_text(), 'loadavg': Path('/proc/loadavg').read_text()}

sources = list((root / 'npu/matmul').rglob('*.cpp')) + list((root / 'npu/matmul').rglob('*.hpp'))
sources += [root / 'npu/vector_add/common.hpp', Path(__file__), root / 'tools/report/build_matmul.py',
            root / 'npu/matmul/CMakeLists.txt', root / 'scripts/benchmark_matmul.sh']
metadata = {
    'backend': 'ttsim', 'measurement_kind': 'host wall time of simulator workload submit + finish',
    'physical_NPU_latency': False, 'date_utc': datetime.now(timezone.utc).isoformat(),
    'tt_metal_tag': 'v0.79.0', 'tt_metal_commit': revision,
    'ttsim_tag': 'v1.11.1', 'ttsim_sha256': sim_hash,
    'host_cpu': subprocess.check_output(['lscpu'], text=True),
    'kernel': subprocess.check_output(['uname', '-r'], text=True).strip(),
    'compiler': subprocess.check_output(['clang++-20', '--version'], text=True).strip(),
    'os_release': Path('/etc/os-release').read_text(), 'host_before': host_state(),
    'sample_count': args.samples, 'warmup_count': 2, 'order_seed': 2026,
    'quantile_method': 'linear interpolation at (n-1)*0.95',
    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'runs': [], 'passed': False,
}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
start = time.perf_counter()
with (out / 'build.log').open('w') as log:
    build = subprocess.run(['python3', str(root / 'tools/report/build_matmul.py')], cwd=root,
                           stdout=log, stderr=subprocess.STDOUT)
metadata['host_build_wall_ms'] = (time.perf_counter() - start) * 1000
if build.returncode:
    metadata['build_exit_code'] = build.returncode
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    raise SystemExit(f'Build failed: {out}/build.log')
binary = root / 'build/matmul/matmul'
metadata['executable_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()

configs = [(v, d, 0) for v in ('baseline', 'reuse_a', 'blocked') for d in (1, 2)]
configs += [('streamed', d, c) for c in (1, 2) for d in (1, 2)]
shapes = [(128, 128, 128), (32, 128, 96), (64, 512, 64), (33, 257, 65)]
jobs = [(shape, config) for shape in shapes for config in configs
        if config[0] != 'blocked' or shape[1] <= 128]
random.Random(2026).shuffle(jobs)

def percentile95(samples):
    ordered = sorted(samples)
    index = (len(ordered) - 1) * 0.95
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

try:
    for index, (shape, (variant, depth, chunk)) in enumerate(jobs):
        mode = f'{variant}_depth{depth}' + (f'_chunk{chunk}' if chunk else '')
        name = '_'.join(map(str, shape)) + '_' + mode
        report_path = out / f'{name}.json'
        fixture = 'fractional' if shape == (33, 257, 65) else 'signed_random'
        selector = fixture + ':' + ','.join(map(str, shape))
        command = [str(binary), str(report_path), variant, str(depth), str(chunk or 2),
                   '--benchmark', str(args.samples), selector]
        start = time.perf_counter()
        with (out / f'{name}.log').open('w') as log:
            log.write('Command: ' + repr(command) + '\n')
            log.flush()
            try:
                code = subprocess.run(command, cwd=root, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired:
                code = 124
            log.write(f'example_exit_code={code}\n')
        process_ms = (time.perf_counter() - start) * 1000
        run = {'order': index, 'shape_mkn': shape, 'mode': mode, 'command': command,
               'report': report_path.name, 'exit_code': code, 'process_wall_ms': process_ms}
        metadata['runs'].append(run)
        if code != 0 or not report_path.exists():
            raise RuntimeError(f'Benchmark failed: {name}, exit {code}')
        report = json.loads(report_path.read_text())
        if not report['passed'] or len(report['cases']) != 1:
            raise RuntimeError('Expected exactly one validated fixture')
        row = report['cases'][0]
        samples = row['timing']['samples_ms']
        if len(samples) != args.samples or not row['timing']['all_repetition_outputs_equal']:
            raise RuntimeError('Invalid repeated timing samples')
        run.update({'median_ms': statistics.median(samples), 'p95_ms': percentile95(samples),
                    'min_ms': min(samples), 'max_ms': max(samples),
                    'mean_ms': statistics.mean(samples), 'stdev_ms': statistics.stdev(samples)})
        print(f'[{index+1}/{len(jobs)}] {shape} {mode}: median={run["median_ms"]:.3f}ms '
              f'p95={run["p95_ms"]:.3f}ms PASS', flush=True)

    # 같은 shape는 필터링된 동일 seed 입력을 사용하며 모든 모드의 출력도 일치해야 합니다.
    comparisons = []
    for shape in shapes:
        runs = [run for run in metadata['runs'] if tuple(run['shape_mkn']) == shape]
        baseline = next(run for run in runs if run['mode'] == 'baseline_depth2')
        reference = json.loads((out / baseline['report']).read_text())['cases'][0]
        for run in runs:
            row = json.loads((out / run['report']).read_text())['cases'][0]
            for key in ('input_fingerprint_a', 'input_fingerprint_b', 'output_fingerprint'):
                if row[key] != reference[key]:
                    raise RuntimeError(f'Controlled comparison mismatch: {key}')
            comparisons.append({**run, 'baseline_depth2_median_ratio': baseline['median_ms'] / run['median_ms'],
                                'read_payload_bytes_model': row['dram_read_payload_model'],
                                'cb_bytes_model': row['cb_capacity_bytes_model'],
                                'read_barriers_model': row['read_barriers_model'],
                                'max_abs_error': row['max_abs_error']})
    summary = {'passed': True, 'kind': metadata['measurement_kind'], 'physical_NPU_latency': False,
               'configurations': len(jobs), 'dispatches': len(jobs) * (args.samples + 2),
               'samples_per_configuration': args.samples, 'comparisons': comparisons}
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# TTSim host dispatch timing', '',
             'Simulator/host wall time; excludes buffer writes/readback, CPU reference and output comparison.',
             'First two submissions are warmup. No physical NPU speedup/latency is measured.', '',
             '| M,K,N | Mode | Median ms | p95 ms | Min–max ms | Baseline depth2 median / median | Read KiB model | CB KiB model |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for run in comparisons:
        lines.append(f'| {run["shape_mkn"]} | {run["mode"]} | {run["median_ms"]:.3f} | '
                     f'{run["p95_ms"]:.3f} | {run["min_ms"]:.3f}–{run["max_ms"]:.3f} | '
                     f'{run["baseline_depth2_median_ratio"]:.2f} | '
                     f'{run["read_payload_bytes_model"]/1024:.1f} | {run["cb_bytes_model"]/1024:.1f} |')
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')
    metadata['passed'] = True
finally:
    metadata['host_after'] = host_state()
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Evidence:', out, flush=True)
raise SystemExit(0 if metadata['passed'] else 1)
