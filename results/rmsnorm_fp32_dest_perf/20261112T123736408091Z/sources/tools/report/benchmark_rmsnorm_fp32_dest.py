"""Matched 3-stage split-versus-FP32-Dest simulator execution-cost comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import median
import subprocess
import time


def load(p):return json.loads(p.read_text())


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=(args.output or root/'results/rmsnorm_fp32_dest_perf'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if (out/'metadata.json').exists():raise SystemExit('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0';sim=metal.parent/'sim/libttsim_wh.so'
    rev=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
    digest=hashlib.sha256(sim.read_bytes()).hexdigest()
    if rev!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
        raise SystemExit('Expected pinned TT-Metal/TTSim')
    env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
             TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
    sources=[p for folder in ('npu/rmsnorm','npu/rmsnorm_fp32_dest') for p in (root/folder).rglob('*')
             if p.is_file() and p.suffix in ('.cpp','.hpp','.txt')]
    sources += [root/'npu/vector_add/common.hpp',Path(__file__),root/'tools/report/rmsnorm_fixtures.py',
                root/'tools/report/build_rmsnorm.py',root/'tools/report/build_rmsnorm_fp32_dest.py',
                root/'scripts/benchmark_rmsnorm_fp32_dest.sh']
    meta=dict(passed=False,tt_metal_revision=rev,ttsim_sha256=digest,depth=2,trials=3,warmups=2,samples=5,
              input_scope='Identical BF16-origin FP32 storage; old split host does not support general FP32 fixtures',
              timing_scope='Host steady_clock, 3-stage enqueue/Finish pipeline; initialization/JIT warmup/IO/validation excluded',
              runs=[],meminfo_before=Path('/proc/meminfo').read_text(),loadavg_before=Path('/proc/loadavg').read_text(),
              source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for p in sources:
        target=out/'sources'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();print('RESULTS='+str(out),flush=True)
    for variant,script in [('split','build_rmsnorm.py'),('compat','build_rmsnorm_fp32_dest.py')]:
        with (out/(variant+'-build.log')).open('w') as log:
            subprocess.run(['python3',str(root/'tools/report'/script)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binaries={'split':root/'build/rmsnorm/rmsnorm','compat':root/'build/rmsnorm_fp32_dest/rmsnorm_fp32_dest'}
    meta['executable_sha256']={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in binaries.items()}
    shutil.copyfile(metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml',sim.parent/'soc_descriptor.yaml')
    subprocess.run(['python3',str(root/'tools/report/rmsnorm_fixtures.py'),'--output',str(out)],check=True)
    all_fixtures=load(out/'fixtures.json')['fixtures']
    keys=['ones_2x32_eps1e-05','signed_65x129_eps1e-05','fractional_17x257_eps1e-05','signed_1x4097_eps1e-05']
    fixtures={f['key']:f for f in all_fixtures if f['key'] in keys and not f['performance']}
    assert len(fixtures)==4
    (out/'selected-fixtures.json').write_text(json.dumps(dict(fixtures=[fixtures[k] for k in keys]),indent=2)+'\n')
    for key in keys:(out/(key+'.txt')).write_text(str(out/fixtures[key]['binary'])+'\n')
    records=[];baseline_outputs={}
    for trial in (1,2,3):
        rng=random.Random(20261112+trial);shapes=keys.copy();rng.shuffle(shapes)
        for key in shapes:
            variants=['split','compat'];rng.shuffle(variants)
            for variant in variants:
                label=f'trial{trial}_{key}_{variant}';report=out/(label+'.json')
                if variant=='split':
                    cmd=[str(binaries[variant]),str(out/(key+'.txt')),str(report),'2','1','unfused','1','--benchmark','2','5']
                else:
                    cmd=[str(binaries[variant]),'run',str(out/(key+'.txt')),str(report),'2','1','compat','--benchmark','2','5']
                start=time.perf_counter()
                with (out/(label+'.log')).open('w') as log:
                    try:code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
                    except subprocess.TimeoutExpired:code=124
                meta['runs'].append(dict(label=label,variant=variant,key=key,trial=trial,command=cmd,exit_code=code,
                    process_wall_seconds=time.perf_counter()-start,loadavg=Path('/proc/loadavg').read_text()));save()
                if code:raise SystemExit('Benchmark failed: '+str(out/(label+'.log')))
                data=load(report);assert data['passed'] and len(data['cases'])==1
                case=data['cases'][0];t=case['benchmark_timing']
                assert case['passed'] and case['repeats']==7 and t['enabled'] and not t['physical_npu_latency']
                assert len(t['samples_ms'])==5 and len(t['warmup_ms'])==2 and len(t['stage_samples_ms'])==5
                for stages,total in zip(t['stage_samples_ms'],t['samples_ms']):
                    assert len(stages)==3 and all(v>0 and math.isfinite(v) for v in stages)
                    assert sum(stages)<=total and total-sum(stages)<1.0
                signature=(case['input_fingerprint'],case['gamma_fingerprint'])
                previous=baseline_outputs.setdefault((variant,key),(signature,case['output_fingerprint'],case['logical_output']))
                assert previous[0]==signature and previous[1]==case['output_fingerprint']
                h,w=case['shape_hw'];ht,wt=(h+31)//32,(w+31)//32
                read_bytes=(5*ht*wt+ht)*4096 if variant=='split' else (4*ht*wt+ht)*4096
                cb_bytes=(6*2+1)*4096 if variant=='split' else (4*2+1)*4096
                records.append(dict(key=key,shape_hw=case['shape_hw'],variant=variant,trial=trial,
                    samples_ms=t['samples_ms'],stage_samples_ms=t['stage_samples_ms'],
                    pipeline_median_ms=median(t['samples_ms']),
                    stage_medians_ms=[median(row[i] for row in t['stage_samples_ms']) for i in range(3)],
                    dram_read_bytes_model=read_bytes,cb_peak_bytes_model=cb_bytes,
                    input_fingerprint=signature[0],gamma_fingerprint=signature[1],output_fingerprint=case['output_fingerprint']))
                print(f'PASS trial{trial} {key} {variant} median_ms={median(t["samples_ms"]):.3f}',flush=True)
    summary=[]
    for key in keys:
        left,right=baseline_outputs['split',key],baseline_outputs['compat',key]
        assert left[0]==right[0] and len(left[2])==len(right[2])
        error=max(abs(a-b) for a,b in zip(left[2],right[2]))
        assert all(abs(a-b)<=1e-4+1e-4*abs(b) for a,b in zip(left[2],right[2]))
        row=dict(key=key,shape_hw=fixtures[key]['shape_hw'],max_abs_output_difference=error)
        for variant in ('split','compat'):
            group=[r for r in records if r['key']==key and r['variant']==variant]
            assert len(group)==3
            trial_medians=[r['pipeline_median_ms'] for r in group]
            row[variant]=dict(median_ms=median(trial_medians),trial_medians_ms=trial_medians,
                trial_min_ms=min(trial_medians),trial_max_ms=max(trial_medians),
                stage_medians_ms=[median(r['stage_medians_ms'][i] for r in group) for i in range(3)],
                dram_read_bytes_model=group[0]['dram_read_bytes_model'],cb_peak_bytes_model=group[0]['cb_peak_bytes_model'])
        row['split_over_compat_ratio']=row['split']['median_ms']/row['compat']['median_ms']
        row['compat_time_change_percent']=(row['compat']['median_ms']/row['split']['median_ms']-1)*100
        summary.append(row)
    (out/'measurements.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(passed=True,depth=2,trials=3,warmups=2,samples=5,
        process_runs=24,timed_rmsnorm_samples=120,rmsnorm_evaluations=168,program_submissions=504,
        results=summary,policy='Host/TTSim execution cost, not physical NPU latency; combined implementation comparison, not isolated unpack-instruction benchmark'),indent=2)+'\n')
    meta['passed']=True;meta['meminfo_after']=Path('/proc/meminfo').read_text();save()
    print('RMSNORM_FP32_DEST_BENCHMARK_ALL_PASS '+str(out),flush=True)


if __name__=='__main__':main()
