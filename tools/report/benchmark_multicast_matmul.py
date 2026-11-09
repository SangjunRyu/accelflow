"""Compare independent/A-only/2D MatMul at matched active-core count using TTSim host timing."""
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
parser.add_argument('--trials', type=int, default=3)
args = parser.parse_args()
if not 5 <= args.samples <= 25 or not 1 <= args.trials <= 5:
    raise SystemExit('Samples must be 5..25; trials must be 1..5')

metal = Path(os.environ.get('TT_METAL_HOME', str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim = Path(os.environ.get('TT_METAL_SIMULATOR', str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
revision = subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest = hashlib.sha256(sim.read_bytes()).hexdigest()
if revision != 'de546d3b146758714d900f11b218c8f9c805f410' or digest != '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TT-Metal v0.79.0 / TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1')
out = root/'results/matmul_multicast_perf'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)

def host_state():
    return {'meminfo':Path('/proc/meminfo').read_text(), 'loadavg':Path('/proc/loadavg').read_text()}

def percentile95(values):
    values = sorted(values)
    pos = (len(values)-1)*0.95
    lo,hi = math.floor(pos),math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)

sources = list((root/'npu/matmul').rglob('*.cpp')) + list((root/'npu/matmul').rglob('*.hpp'))
sources += [Path(__file__), root/'scripts/benchmark_multicast_matmul.sh',
            root/'tools/report/build_matmul.py', root/'npu/matmul/CMakeLists.txt', root/'npu/vector_add/common.hpp']
metadata = {
    'date_utc':datetime.now(timezone.utc).isoformat(), 'backend':'ttsim',
    'measurement_kind':'host steady_clock: EnqueueMeshWorkload + Finish',
    'physical_chip_latency':False, 'tt_metal_revision':revision, 'ttsim_sha256':digest,
    'samples_per_process':args.samples, 'warmups_per_process':2, 'independent_trials':args.trials,
    'order_seed_base':2026, 'host_before':host_state(),
    'os_release':Path('/etc/os-release').read_text(),
    'cpu':subprocess.check_output(['lscpu'],text=True),
    'compiler':subprocess.check_output(['clang++-20','--version'],text=True),
    'kernel':subprocess.check_output(['uname','-r'],text=True).strip(),
    'scope_note':'Existing implementations differ in partition and reader NoC/RISC-V placement; not an isolated multicast ablation',
    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'runs':[], 'passed':False,
}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
started = time.perf_counter()
with (out/'build.log').open('w') as log:
    code = subprocess.run(['python3',str(root/'tools/report/build_matmul.py')],cwd=root,
                          stdout=log,stderr=subprocess.STDOUT).returncode
metadata['host_build_wall_ms'] = (time.perf_counter()-started)*1000
metadata['build_exit_code'] = code
if code:
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    raise SystemExit(f'Build failed: {out}/build.log')
binary = root/'build/matmul/matmul'
metadata['executable_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()

# 세 전략 모두 실제 4코어인 입력만 주 비교에 사용합니다.
shapes = [(128,128,128), (96,64,160), (65,257,97), (32,64,128)]
configs = [('multicore',depth,1) for depth in (1,2)]
configs += [(variant,depth,4) for variant in ('multicore','mcast','mcast2d') for depth in (1,2)]
results, fingerprints = {}, {}
total = len(shapes)*len(configs)*args.trials

def inspect_model(row, variant, requested, depth):
    mt,kt,nt = row['tile_grid_mkn']
    dist = row['distribution']
    active = row['core_count']
    a_calls = b_calls = a_delivered = b_delivered = 0
    if variant == 'multicore':
        expected_active = min(requested,mt*nt)
        a_reads = b_reads = mt*kt*nt
        ids = []
        counts = []
        for assignment in dist['assignments']:
            start,count = assignment['start_tile'],assignment['tile_count']
            ids += list(range(start,start+count))
            counts.append(count)
        topology = 'contiguous C tile ranges'
        reader_noc = 'RISCV_1 / NoC1'
    else:
        if variant == 'mcast':
            rows,cols = 1,min(requested,nt)
        else:
            rows,cols = (1,min(4,nt)) if mt == 1 else (2,2)
            if dist['core_grid_rows_cols'] != [rows,cols]:
                raise RuntimeError('Unexpected 2D grid')
        expected_active = rows*cols
        a_reads = mt*kt*((nt+cols-1)//cols)
        b_reads = nt*kt*((mt+rows-1)//rows)
        a_calls = a_reads if cols>1 else 0
        b_calls = b_reads if rows>1 else 0
        a_delivered,b_delivered = a_reads*(cols-1),b_reads*(rows-1)
        for key,value in [('a_dram_reads_model',a_reads), ('a_multicast_calls_model',a_calls),
                          ('a_receiver_deliveries_model',a_delivered)]:
            if dist[key] != value:
                raise RuntimeError(f'Multicast model mismatch: {key}')
        if variant == 'mcast2d':
            if dist['b_dram_reads_model'] != b_reads or dist['b_multicast_calls_model'] != b_calls or dist['b_receiver_deliveries_model'] != b_delivered:
                raise RuntimeError('B multicast model mismatch')
        ids = [tile for item in dist['assignments'] for tile in item['output_tile_ids']]
        counts = [len(item['output_tile_ids']) for item in dist['assignments']]
        topology = f'{rows} rows x {cols} cols, strided C'
        reader_noc = 'RISCV_0 / NoC0'
    if active != expected_active or dist['active_cores'] != expected_active or len(counts) != active:
        raise RuntimeError('Active core count mismatch')
    if requested == 4 and active != 4:
        raise RuntimeError('Primary comparison must use exactly four active cores')
    if sorted(ids) != list(range(mt*nt)):
        raise RuntimeError('Gap/overlap in output ownership')
    expected = {'a_tile_reads_model':a_reads, 'b_tile_reads_model':b_reads,
                'dram_read_payload_model':2048*(a_reads+b_reads),
                'dram_write_payload_model':2048*mt*nt, 'matmul_tile_calls_model':mt*kt*nt,
                'read_barriers_model':a_reads+b_reads,
                'cb_capacity_bytes_model':3*depth*2048,
                'cb_capacity_bytes_total_model':active*3*depth*2048}
    for key,value in expected.items():
        if row[key] != value:
            raise RuntimeError(f'Report model mismatch: {key}')
    return {'active_cores':active, 'topology':topology, 'reader_placement':reader_noc,
            'output_tiles_per_core':counts, 'max_matmul_calls_per_core_model':max(counts)*kt,
            'read_bytes_model':2048*(a_reads+b_reads), 'write_bytes_model':2048*mt*nt,
            'receiver_data_bytes_model':2048*(a_delivered+b_delivered),
            'payload_multicast_calls_model':a_calls+b_calls,
            'ready_atomic_increments_model':a_delivered+b_delivered,
            'valid_multicast_calls_model':a_calls+b_calls,
            'input_transport_write_barriers_model':2*(a_calls+b_calls),
            'cb_bytes_per_core_model':3*depth*2048, 'cb_bytes_total_model':active*3*depth*2048}

try:
    for trial in range(args.trials):
        jobs = [(shape,config) for shape in shapes for config in configs]
        random.Random(2026+trial).shuffle(jobs)
        for shape,(variant,depth,requested) in jobs:
            mode = f'{variant}_cores{requested}_depth{depth}'
            name = '_'.join(map(str,shape))+'_'+mode+f'_trial{trial+1}'
            path = out/f'{name}.json'
            command = [str(binary),str(path),variant,str(depth),str(requested),'--benchmark',
                       str(args.samples),'signed_random:'+','.join(map(str,shape))]
            started = time.perf_counter()
            with (out/f'{name}.log').open('w') as log:
                log.write('Command: '+repr(command)+'\n');log.flush()
                try:
                    code = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
                except subprocess.TimeoutExpired:
                    code = 124
                log.write(f'example_exit_code={code}\n')
            run = {'order':len(metadata['runs']), 'trial':trial+1, 'shape_mkn':shape,
                   'variant':variant, 'depth':depth, 'requested_cores':requested, 'mode':mode,
                   'report':path.name, 'command':command, 'exit_code':code,
                   'whole_process_wall_ms':(time.perf_counter()-started)*1000,
                   'host_loadavg_after':Path('/proc/loadavg').read_text()}
            metadata['runs'].append(run)
            if code or not path.is_file():
                raise RuntimeError(f'Failed {name}; see full log')
            report = json.loads(path.read_text())
            if not report['passed'] or len(report['cases']) != 1:
                raise RuntimeError('Expected one validated fixture')
            row = report['cases'][0]
            if not all(row[key] for key in ('passed','padding_verified','output_sentinel_overwritten',
                                            'layout_matches_official_tilizer','fp32_destination_accumulation')):
                raise RuntimeError('Functional contract failed')
            values = tuple(row[key] for key in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint'))
            if shape in fingerprints and fingerprints[shape] != values:
                raise RuntimeError('Controlled input/output comparison differs')
            fingerprints[shape] = values
            timing = row['timing']
            samples = timing['samples_ms']
            if len(samples) != args.samples or len(timing['warmup_ms']) != 2 or not timing['all_repetition_outputs_equal']:
                raise RuntimeError('Invalid repeated timing samples')
            model = inspect_model(row,variant,requested,depth)
            run['median_ms'] = statistics.median(samples)
            results.setdefault((shape,mode),[]).append({'run':run,'samples':samples,'model':model})
            print(f'[{len(metadata["runs"])}/{total}] trial{trial+1} {shape} {mode}: '
                  f'median={run["median_ms"]:.3f}ms PASS',flush=True)

    comparisons = []
    for shape in shapes:
        for variant,depth,requested in configs:
            mode = f'{variant}_cores{requested}_depth{depth}'
            items = results[(shape,mode)]
            medians = [item['run']['median_ms'] for item in items]
            samples = [sample for item in items for sample in item['samples']]
            reference = results[(shape,f'multicore_cores4_depth{depth}')]
            reference_median = statistics.median([item['run']['median_ms'] for item in reference])
            paired = [a['run']['median_ms']/b['run']['median_ms'] for a,b in zip(reference,items)]
            primary = statistics.median(medians)
            comparisons.append({
                'shape_mkn':shape, 'variant':variant, 'depth':depth, 'requested_cores':requested,
                'mode':mode, **items[0]['model'],
                'median_of_trial_medians_ms':primary, 'trial_medians_ms':medians,
                'pooled_p95_ms':percentile95(samples), 'pooled_median_ms':statistics.median(samples),
                'min_ms':min(samples), 'max_ms':max(samples), 'sample_count':len(samples),
                'independent4_median_ratio':reference_median/primary,
                'within_trial_independent4_ratios':paired,
                'trial_median_min_ms':min(medians), 'trial_median_max_ms':max(medians),
                'source_reports':[item['run']['report'] for item in items]})
    summary = {'passed':True, 'unique_configurations':len(shapes)*len(configs),
               'process_trials':total, 'timed_samples':total*args.samples, 'dispatches':total*(args.samples+2),
               'scope':metadata['measurement_kind'], 'physical_chip_latency':False,
               'primary_statistic':'median of independent process trial medians',
               'comparisons':comparisons,
               'interpretation':'Different partition/reader placement and communication; measured whole implementations, not multicast alone'}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# Independent vs A-only vs 2D multicast: TTSim host timing','',
             '| M,K,N | Mode | Active | Trial median ms | Trials ms | p95 ms | Independent4 / current | DRAM read KiB model | Receiver data KiB model | Max matmul calls/core model |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in comparisons:
        trials='/'.join(f'{value:.2f}' for value in r['trial_medians_ms'])
        lines.append(f'| {r["shape_mkn"]} | {r["mode"]} | {r["active_cores"]} | '
                     f'{r["median_of_trial_medians_ms"]:.3f} | {trials} | {r["pooled_p95_ms"]:.3f} | '
                     f'{r["independent4_median_ratio"]:.2f} | {r["read_bytes_model"]/1024:.1f} | '
                     f'{r["receiver_data_bytes_model"]/1024:.1f} | {r["max_matmul_calls_per_core_model"]} |')
    lines += ['', 'Host/simulator wall time. Source traffic counts are models, not runtime counters or physical hardware speedup.', '']
    (out/'comparison.md').write_text('\n'.join(lines))
    metadata['passed'] = True
finally:
    metadata['host_after'] = host_state()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
