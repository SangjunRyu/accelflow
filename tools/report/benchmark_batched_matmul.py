"""Validate K-batched multicast and compare TTSim host dispatch times (not chip latency)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--samples', type=int, default=5)
args = parser.parse_args()
if not 1 <= args.trials <= 5 or not 5 <= args.samples <= 25:
    raise SystemExit('trials 1..5, samples 5..25')
metal = Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
sim = Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'
revision = subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
sim_hash = hashlib.sha256(sim.read_bytes()).hexdigest()
if revision != 'de546d3b146758714d900f11b218c8f9c805f410' or sim_hash != '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TT-Metal v0.79.0 / TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1')
out = root/'results/matmul_batched_perf'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
print(out, flush=True)
sources = list((root/'npu/matmul').rglob('*.cpp'))+list((root/'npu/matmul').rglob('*.hpp'))
sources += [Path(__file__),root/'tools/report/build_matmul.py',root/'npu/vector_add/common.hpp']
meta = dict(tt_metal_revision=revision, ttsim_sha256=sim_hash, samples=args.samples,
            trials=args.trials, warmups=2, measurement='host EnqueueMeshWorkload + Finish; not chip latency',
            meminfo_before=Path('/proc/meminfo').read_text(),
            source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            runs=[], passed=False)
def save():
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
save()
with (out/'build.log').open('w') as log:
    subprocess.run(['python3',str(root/'tools/report/build_matmul.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
binary = root/'build/matmul/matmul'
meta['executable_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
fingerprints = {}
def run(label, variant, depth, cores, shape=None):
    report = out/(label+'.json')
    cmd=[str(binary),str(report),variant,str(depth),str(cores)]
    if shape:
        cmd+=['--benchmark',str(args.samples),'signed_random:'+','.join(map(str,shape))]
    with (out/(label+'.log')).open('w') as log:
        proc=subprocess.run(cmd,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    meta['runs'].append(dict(label=label,command=cmd,exit_code=proc.returncode))
    save()
    if proc.returncode:
        raise SystemExit('Failed: '+str(out/(label+'.log')))
    data=json.loads(report.read_text())
    assert data['passed'] and data['cases']
    for row in data['cases']:
        assert row['passed'] and row['padding_verified'] and row['output_sentinel_overwritten']
        key=(row['name'],tuple(row['shape_mkn']))
        fp=tuple(row[k] for k in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint'))
        if key in fingerprints:
            assert fingerprints[key]==fp, ('Different input/output',label,key)
        fingerprints[key]=fp
        if row['B_multicast_enabled']:
            mt,kt,nt=row['tile_grid_mkn']
            dist=row['distribution']; rows,cols=dist['core_grid_rows_cols']
            batch=data['multicast_batch_tiles']; chunks=(kt+batch-1)//batch
            a=mt*kt*((nt+cols-1)//cols); b=nt*kt*((mt+rows-1)//rows)
            ac=a//kt*chunks if cols>1 else 0
            bc=b//kt*chunks if rows>1 else 0
            assert dist['a_multicast_calls_model']==ac and dist['b_multicast_calls_model']==bc
            assert row['a_tile_reads_model']==a and row['b_tile_reads_model']==b
            assert row['cb_capacity_bytes_model']==2048*depth*(2*batch+1)
    print('PASS '+label+' cases='+str(len(data['cases'])),flush=True)
    return data

# Small/tail/fractional/large-K fixtures, depth wrap and 1/2/4-core fallback.
for batch in (1,4,8):
    for depth in (1,2):
        run(f'correctness_b{batch}_d{depth}_p4',f'mcast2d_batch{batch}',depth,4)
for cores in (1,2):
    for depth in (1,2):
        run(f'correctness_b8_d{depth}_p{cores}','mcast2d_batch8',depth,cores)

configs=['multicore','mcast','mcast2d','mcast2d_batch1','mcast2d_batch4','mcast2d_batch8']
shapes=[(320,320,320),(640,320,640)]
records=[]
for trial in range(args.trials):
    jobs=[(shape,config) for shape in shapes for config in configs]
    random.Random(2026+trial).shuffle(jobs)
    for shape,config in jobs:
        label=f'trial{trial+1}_'+('x'.join(map(str,shape)))+'_'+config
        data=run(label,config,2,4,shape)
        row=data['cases'][0]
        records.append(dict(shape=shape,variant=config,trial=trial+1,
                            median_ms=statistics.median(row['timing']['samples_ms']),case=row))
        (out/'measurements.json').write_text(json.dumps(records,indent=2)+'\n')
summary=[]
for shape in shapes:
    for config in configs:
        rr=[x for x in records if tuple(x['shape'])==shape and x['variant']==config]
        medians=[x['median_ms'] for x in rr]
        row=rr[0]['case']
        summary.append(dict(shape=shape,variant=config,median_ms=statistics.median(medians),
                            trial_medians_ms=medians,dram_read_bytes=row['dram_read_payload_model'],
                            cb_bytes_per_core=row['cb_capacity_bytes_model'],
                            distribution=row['distribution']))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
meta['passed']=True
meta['meminfo_after']=Path('/proc/meminfo').read_text()
meta['correctness_cases']=sum(len(json.loads((out/(r['label']+'.json')).read_text())['cases']) for r in meta['runs'] if r['label'].startswith('correctness'))
save()
for row in summary:
    print(row['shape'],row['variant'],round(row['median_ms'],3),row['trial_medians_ms'],flush=True)
print('ALL_PASS '+str(out),flush=True)
