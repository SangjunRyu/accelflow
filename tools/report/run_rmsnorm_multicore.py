"""Validate disjoint row-group partitioning for 1/2/4-core fused RMSNorm."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time


def read_json(path):
    return json.loads(path.read_text())


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 65535


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = (args.output or root/'results/rmsnorm_multicore'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if (out/'metadata.json').exists():
        raise SystemExit('Use a new output directory')
    scratch = out/'scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for name,h,w in [('balanced_narrow',256,33),('balanced',256,257),('wide',128,1025),
                     ('tail_mixed',65,65),('idle_zero',1,33)]:
        hp,wp = (h+31)//32*32,(w+31)//32*32
        bits=[]
        for row in range(hp):
            for col in range(wp):
                value = ((row*13+col*7)%31-15)/8 if row<h and col<w else 0.0
                if name=='tail_mixed' and row<h and col<w:
                    value=[0.0001,1.0,32.0,-32.0,-0.001][(row+col)%5]
                if name=='idle_zero':value=0.0
                bits.append(bf16(value))
        gamma=[bf16((col%9-4)/4 if col<w else 0.0) for col in range(wp)]
        key=f'{name}_{h}x{w}'
        binary=scratch/(key+'.bin')
        binary.write_bytes(struct.pack('<8sIIIIf',b'AFRM0001',h,w,hp,wp,1e-5)+
                           struct.pack('<'+'H'*(len(bits)+len(gamma)),*(bits+gamma)))
        fixtures.append(dict(key=key,shape_hw=[h,w],binary='scratch/'+binary.name,
                             sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    (out/'fixtures.json').write_text(json.dumps(dict(passed=True,fixtures=fixtures),indent=2)+'\n')
    full=out/'fixtures.txt';pilot=out/'pilot.txt'
    full.write_text(''.join(str(out/f['binary'])+'\n' for f in fixtures))
    pilot.write_text(''.join(str(out/f['binary'])+'\n' for f in fixtures[-2:]))
    metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
    sim=metal.parent/'sim/libttsim_wh.so'
    revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
    digest=hashlib.sha256(sim.read_bytes()).hexdigest()
    if revision!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
        raise SystemExit('Expected pinned TT-Metal/TTSim')
    (sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
    sources=list((root/'npu/rmsnorm').rglob('*.cpp'))+[root/'npu/rmsnorm/CMakeLists.txt',
             root/'npu/vector_add/common.hpp',root/'tools/report/build_rmsnorm.py',Path(__file__)]
    meta=dict(passed=False,tt_metal_revision=revision,ttsim_sha256=digest,runs=[],
              meminfo_before=Path('/proc/meminfo').read_text(),
              source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for source in sources:
        target=out/'sources'/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save()
    print('RESULTS='+str(out),flush=True)
    with (out/'build.log').open('w') as log:
        subprocess.run(['python3',str(root/'tools/report/build_rmsnorm.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=root/'build/rmsnorm/rmsnorm';meta['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
    def run(label,cores,inputs,repeats):
        command=[str(exe),str(inputs),str(out/(label+'.json')),'2',str(repeats),'fused',str(cores)]
        start=time.perf_counter()
        with (out/(label+'.log')).open('w') as log:
            try:code=subprocess.run(command,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
            except subprocess.TimeoutExpired:code=124
        meta['runs'].append(dict(label=label,command=command,exit_code=code,process_wall_seconds=time.perf_counter()-start));save()
        if code:raise SystemExit('Multicore RMSNorm failed: '+str(out/(label+'.log')))
        data=read_json(out/(label+'.json'));assert data['passed']
        print('PASS '+label,flush=True)
        return data
    run('pilot-1core',1,pilot,1)
    run('pilot-4core',4,pilot,1)
    results={n:run(f'cores{n}',n,full,1) for n in (1,2,4)}
    repeated=run('repeat-tail-4core',4,pilot,2)
    reference={c['key']:c for c in results[1]['cases']}
    comparisons=[]
    for cores,data in results.items():
        assert len(data['cases'])==5
        for case in data['cases']:
            base=reference[case['key']];h,w=case['shape_hw'];ht,wt=(h+31)//32,(w+31)//32
            active=min(cores,ht);coverage=[0]*ht
            assert case['active_cores']==active and case['idle_requested_cores']==cores-active
            assert len(case['shards'])==active
            for shard in case['shards']:
                start,count=shard['row_group_start'],shard['row_group_count']
                assert count>0 and shard['tile_start']==start*wt and shard['tile_count']==count*wt
                for group in range(start,start+count):coverage[group]+=1
            assert coverage==[1]*ht
            assert max(s['row_group_count'] for s in case['shards'])-min(s['row_group_count'] for s in case['shards'])<=1
            assert case['dram_payload_bytes_model']==3*ht*wt*4096
            assert case['cb_peak_payload_bytes_model']==base['cb_peak_payload_bytes_model']
            assert case['cb_total_payload_bytes_model']==case['cb_peak_payload_bytes_model']*active
            for name in ['input_fingerprint','gamma_fingerprint','output_fingerprint']:assert case[name]==base[name]
            assert struct.pack('<'+'f'*len(case['logical_output']),*case['logical_output'])==struct.pack('<'+'f'*len(base['logical_output']),*base['logical_output'])
            comparisons.append(dict(key=case['key'],requested_cores=cores,active_cores=active,
                row_groups_per_core=[s['row_group_count'] for s in case['shards']],
                per_core_cb_bytes=case['cb_peak_payload_bytes_model'],total_cb_bytes=case['cb_total_payload_bytes_model'],
                dram_payload_bytes_model=case['dram_payload_bytes_model'],bitwise_equal_to_single_core=True))
    for case in repeated['cases']:assert case['output_fingerprint']==reference[case['key']]['output_fingerprint']
    summary=dict(passed=True,unique_configurations=15,full_suite_submissions=15,repeated_tail_submissions=4,
                 pilot_submissions=4,total_operator_submissions=23,depth=2,bitwise_equal_across_core_counts=True,
                 max_abs_error=max(c['max_abs_error'] for d in results.values() for c in d['cases']),
                 comparisons=comparisons,performance_policy='Process wall time is simulator/JIT/host cost, not real NPU speedup')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    meta['passed']=True;save();print('MULTICORE_RMSNORM_ALL_PASS '+str(out),flush=True)


if __name__=='__main__':main()
