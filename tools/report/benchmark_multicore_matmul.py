"""Repeated-process TTSim multicore timing and controlled workload/storage comparison."""
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
    raise SystemExit('Expected samples=5..25 and independent process trials=1..5')

metal = Path(os.environ.get('TT_METAL_HOME', str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim = Path(os.environ.get('TT_METAL_SIMULATOR', str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
revision = subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest = hashlib.sha256(sim.read_bytes()).hexdigest()
if revision != 'de546d3b146758714d900f11b218c8f9c805f410' or digest != '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TT-Metal v0.79.0 and TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1')
out = root/'results/matmul_multicore_perf'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)

def host_state():
    return {'meminfo':Path('/proc/meminfo').read_text(), 'loadavg':Path('/proc/loadavg').read_text()}

def p95(values):
    values = sorted(values)
    pos = (len(values)-1)*0.95
    lo,hi = math.floor(pos),math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)

sources = list((root/'npu/matmul').rglob('*.cpp')) + list((root/'npu/matmul').rglob('*.hpp'))
sources += [Path(__file__),root/'scripts/benchmark_multicore_matmul.sh',
            root/'tools/report/build_matmul.py',root/'npu/matmul/CMakeLists.txt',root/'npu/vector_add/common.hpp']
metadata = {
    'date_utc':datetime.now(timezone.utc).isoformat(), 'backend':'ttsim',
    'measurement_kind':'host steady_clock: EnqueueMeshWorkload + Finish',
    'physical_chip_latency':False, 'tt_metal_revision':revision,'ttsim_sha256':digest,
    'samples_per_process':args.samples,'warmups_per_process':2,'independent_trials':args.trials,
    'order_seed_base':2026,'clock_summary':'median of independent trial medians; pooled descriptive p95',
    'host_before':host_state(),'os_release':Path('/etc/os-release').read_text(),
    'cpu':subprocess.check_output(['lscpu'],text=True),
    'compiler':subprocess.check_output(['clang++-20','--version'],text=True),
    'kernel':subprocess.check_output(['uname','-r'],text=True).strip(),
    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'runs':[],'passed':False,
}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
build_start = time.perf_counter()
with (out/'build.log').open('w') as log:
    code = subprocess.run(['python3',str(root/'tools/report/build_matmul.py')],cwd=root,
                          stdout=log,stderr=subprocess.STDOUT).returncode
metadata['host_build_wall_ms']=(time.perf_counter()-build_start)*1000
metadata['build_exit_code']=code
if code:
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    raise SystemExit(f'Build failed: {out}/build.log')
binary=root/'build/matmul/matmul'
metadata['executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()

# 네 입력: 균등 분배, 나머지 분배, 큰 K, 출력 한 타일인 대조군입니다.
fixtures = [('signed_random',(128,128,128)), ('signed_random',(64,64,96)),
            ('signed_random',(64,512,64)), ('deterministic',(2,3,2))]
configs=[('baseline',d,1) for d in (1,2)]
configs += [('multicore',d,c) for c in (1,2,4) for d in (1,2)]
expected_reference={}
results={}
total=len(fixtures)*len(configs)*args.trials
try:
    for trial in range(args.trials):
        jobs=[(fixture,config) for fixture in fixtures for config in configs]
        random.Random(2026+trial).shuffle(jobs)
        for (fixture,shape),(variant,depth,requested) in jobs:
            mode=f'{variant}_cores{requested}_depth{depth}'
            key=(shape,mode)
            name='_'.join(map(str,shape))+'_'+mode+f'_trial{trial+1}'
            report_path=out/f'{name}.json'
            selector=fixture+':'+','.join(map(str,shape))
            command=[str(binary),str(report_path),variant,str(depth),str(requested if variant=='multicore' else 2),
                     '--benchmark',str(args.samples),selector]
            started=time.perf_counter()
            with (out/f'{name}.log').open('w') as log:
                log.write('Command: '+repr(command)+'\n');log.flush()
                try:
                    code=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
                except subprocess.TimeoutExpired:
                    code=124
                log.write(f'example_exit_code={code}\n')
            run={'order':len(metadata['runs']),'trial':trial+1,'fixture':fixture,'shape_mkn':shape,
                 'mode':mode,'exit_code':code,'report':report_path.name,'command':command,
                 'whole_process_wall_ms':(time.perf_counter()-started)*1000,
                 'host_loadavg_after':Path('/proc/loadavg').read_text()}
            metadata['runs'].append(run)
            if code or not report_path.is_file():
                raise RuntimeError(f'Failed {name}; check .log')
            report=json.loads(report_path.read_text())
            if not report['passed'] or len(report['cases'])!=1:
                raise RuntimeError('Expected one CPU-validated fixture')
            row=report['cases'][0]
            if not all(row[field] for field in ('passed','padding_verified','output_sentinel_overwritten','layout_matches_official_tilizer','fp32_destination_accumulation')):
                raise RuntimeError('Functional contract failed')
            fingerprints=tuple(row[field] for field in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint'))
            if shape in expected_reference and fingerprints!=expected_reference[shape]:
                raise RuntimeError('Inputs or outputs differ across configurations/trials')
            expected_reference[shape]=fingerprints
            samples=row['timing']['samples_ms']
            if len(samples)!=args.samples or not row['timing']['all_repetition_outputs_equal']:
                raise RuntimeError('Invalid timing samples')
            mt,kt,nt=row['tile_grid_mkn']
            tiles=mt*nt
            active=min(requested,tiles)
            if row['core_count']!=active or row['cb_capacity_bytes_total_model']!=active*3*depth*2048:
                raise RuntimeError('Core or CB model mismatch')
            if row['dram_read_payload_model']!=4096*tiles*kt:
                raise RuntimeError('Traffic model differs from baseline')
            if variant=='multicore':
                assignments=row['distribution']['assignments']
                starts=0
                for item in assignments:
                    if item['start_tile']!=starts or item['tile_count']<=0:
                        raise RuntimeError('Gap or overlap in core assignment')
                    starts+=item['tile_count']
                if starts!=tiles or len(assignments)!=active:
                    raise RuntimeError('Incomplete core coverage')
                max_tiles=max(a['tile_count'] for a in assignments)
            else:
                max_tiles=tiles
            run['median_ms']=statistics.median(samples)
            run['p95_ms']=p95(samples)
            results.setdefault(key,[]).append({'run':run,'row':row,'samples':samples,'max_tiles':max_tiles})
            print(f'[{len(metadata["runs"])}/{total}] trial{trial+1} {shape} {mode}: '
                  f'median={run["median_ms"]:.3f}ms PASS',flush=True)

    comparison=[]
    for shape in [shape for _,shape in fixtures]:
        for variant,depth,requested in configs:
            mode=f'{variant}_cores{requested}_depth{depth}'
            items=results[(shape,mode)]
            medians=[item['run']['median_ms'] for item in items]
            samples=[sample for item in items for sample in item['samples']]
            row=items[0]['row']
            same_depth_base=results[(shape,f'baseline_cores1_depth{depth}')]
            same_depth_mc1=results[(shape,f'multicore_cores1_depth{depth}')]
            representative=statistics.median(medians)
            mt,kt,nt=row['tile_grid_mkn']
            max_tiles=items[0]['max_tiles']
            comparison.append({
                'shape_mkn':shape,'mode':mode,'requested_cores':requested,'active_cores':row['core_count'],
                'depth':depth,'median_of_trial_medians_ms':representative,'trial_medians_ms':medians,
                'pooled_median_ms':statistics.median(samples),'pooled_p95_ms':p95(samples),
                'min_ms':min(samples),'max_ms':max(samples),'sample_count':len(samples),
                'same_depth_baseline_median_ratio':statistics.median([x['run']['median_ms'] for x in same_depth_base])/representative,
                'same_depth_multicore1_median_ratio':statistics.median([x['run']['median_ms'] for x in same_depth_mc1])/representative,
                'read_payload_bytes_model':row['dram_read_payload_model'],
                'cb_bytes_per_core_model':row['cb_capacity_bytes_model'],
                'cb_bytes_total_model':row['cb_capacity_bytes_total_model'],
                'output_tiles':mt*nt,'max_output_tiles_per_core':max_tiles,
                'max_matmul_calls_per_core_model':max_tiles*kt,
                'work_distribution_factor_model':mt*nt/max_tiles,
                'source_reports':[item['run']['report'] for item in items],
            })
    summary={'passed':True,'unique_configurations':len(fixtures)*len(configs),
             'process_trials':total,'timed_samples':total*args.samples,'dispatches':total*(args.samples+2),
             'scope':metadata['measurement_kind'],'physical_chip_latency':False,'comparisons':comparison,
             'primary_statistic':'median of independent process trial medians',
             'pooled_p95_policy':'descriptive linear interpolation; samples in a process are correlated, no confidence claim'}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Independent multicore: measured TTSim/host execution cost','',
           '| M,K,N | Mode | Active cores | Median of trials ms | Trial medians ms | Pooled p95 ms | MC1 / current, same depth | CB total KiB | Work distribution factor model |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in comparison:
        trials='/'.join(f'{x:.2f}' for x in r['trial_medians_ms'])
        lines.append(f'| {r["shape_mkn"]} | {r["mode"]} | {r["active_cores"]} | '
                     f'{r["median_of_trial_medians_ms"]:.3f} | {trials} | {r["pooled_p95_ms"]:.3f} | '
                     f'{r["same_depth_multicore1_median_ratio"]:.2f} | {r["cb_bytes_total_model"]/1024:.0f} | '
                     f'{r["work_distribution_factor_model"]:.2f} |')
    lines+=['','Ratios compare host/simulator elapsed time. Work distribution is a source model, not hardware speedup.','']
    (out/'comparison.md').write_text('\n'.join(lines))
    metadata['passed']=True
finally:
    metadata['host_after']=host_state()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
