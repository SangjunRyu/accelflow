"""Validate three CUDA row-sum implementations, then aggregate repeated timings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--executable', type=Path, required=True)
p.add_argument('--local-fixtures', type=Path, required=True)
p.add_argument('--trials', type=int, default=3)
args = p.parse_args()
if not 1 <= args.trials <= 10:
    p.error('trials must be 1..10')
out = args.output
manifest = json.loads((out / 'fixtures.json').read_text(encoding='utf-8-sig'))
assert manifest['passed']
variants = ['naive', 'tree', 'shuffle']
jobs = [(f, v, trial) for f in manifest['fixtures'] for v in variants
        for trial in range(1, (args.trials if f['performance'] else 1) + 1)]
random.Random(2026).shuffle(jobs)
measurements = []
for fixture, variant, trial in jobs:
    binary = args.local_fixtures / Path(fixture['binary']).name
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == fixture['sha256']
    key = f"{fixture['key']}_{variant}_trial{trial}"
    report = out / (key + '.json')
    samples, batch = (15, 20) if fixture['performance'] else (5, 1)
    command = [str(args.executable), str(binary), str(report), variant, str(samples), str(batch)]
    with (out / (key + '.log')).open('w', encoding='utf-8') as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if proc.returncode:
        raise SystemExit('CUDA failed: ' + key)
    data = json.loads(report.read_text())
    assert data['passed'] and data['validated_outputs'] == 1 + 2 * samples
    if 'npu_row_sums' in fixture:
        for got, expected in zip(data['row_sums'], fixture['npu_row_sums'], strict=True):
            assert abs(got - expected) <= (0 if data['integer_exact'] else 1e-4 + 1e-5 * abs(expected))
    measurements.append(dict(key=fixture['key'], variant=variant, trial=trial, performance=fixture['performance'],
                             command=command, report=report.name, exit_code=proc.returncode,
                             npu_compared='npu_row_sums' in fixture,
                             max_abs_error=data['max_abs_error'], validated_outputs=data['validated_outputs'],
                             kernel_ms=statistics.median(data['kernel_event_batch_average_ms']),
                             pipeline_ms=statistics.median(data['host_pipeline_ms'])))
    (out / 'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n', encoding='utf-8')
    print('PASS ' + key, flush=True)
rows = []
for fixture in manifest['fixtures']:
    if not fixture['performance']:
        continue
    row = dict(key=fixture['key'], variants={})
    for variant in variants:
        runs = [x for x in measurements if x['key'] == fixture['key'] and x['variant'] == variant]
        row['variants'][variant] = dict(kernel_ms=statistics.median(x['kernel_ms'] for x in runs),
                                        pipeline_ms=statistics.median(x['pipeline_ms'] for x in runs),
                                        process_medians_ms=[x['kernel_ms'] for x in runs])
    rows.append(row)
summary = dict(passed=True, timestamp_utc=datetime.now(timezone.utc).isoformat(), processes=len(measurements),
               npu_input_cases=sum('npu_row_sums' in f for f in manifest['fixtures']),
               cuda_only_performance_cases=sum(f['performance'] for f in manifest['fixtures']), trials=args.trials,
               validated_outputs=sum(x['validated_outputs'] for x in measurements),
               max_abs_error=max(x['max_abs_error'] for x in measurements), rows=rows,
               executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
               timing='median of process medians; CUDA Event batch average; pipeline includes padded FP32 transfers',
               limitations=['No bank-conflict hardware counters', 'No real NPU performance comparison',
                            'Fixed 256 threads; not auto-tuned', 'Different summation orders; finite BF16 input tests only'])
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(f"CUDA_REDUCTION_PASS processes={len(measurements)} validated_outputs={summary['validated_outputs']}")
