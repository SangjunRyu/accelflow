"""Compare row- and column-partitioned RMSNorm on the pinned Wormhole simulator."""
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


def load(path):
    return json.loads(path.read_text())


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 65535


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = (args.output or root/'results/rmsnorm_column'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if (out/'metadata.json').exists():
        raise SystemExit('Use a new output directory')
    scratch = out/'scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    fixtures = []
    shapes = [('idle_zero',1,1,1e-5), ('two_tiles',17,33,1e-5),
              ('uneven',33,129,1e-5), ('remote_spike',33,257,1e-6),
              ('mixed',65,65,1e-5), ('wide',1,1025,1e-5), ('epsilon',1,31,1e-3)]
    for name,h,w,eps in shapes:
        hp,wp = (h+31)//32*32,(w+31)//32*32
        bits=[]
        for row in range(hp):
            for col in range(wp):
                value = ((row*13+col*7)%31-15)/8
                if name=='uneven': value=((row*997+col*73)%1009-504)/137
                if name=='mixed': value=[0.0001,1.0,32.0,-32.0,-0.001][(row+col)%5]
                if name=='remote_spike': value=32.0 if col==w-1 else 1.0
                if name=='epsilon': value=0.0001 if col%2 else -0.0001
                if name=='idle_zero' or row>=h or col>=w: value=0.0
                bits.append(bf16(value))
        gamma=[bf16((col%9-4)/4 if col<w else 0.0) for col in range(wp)]
        key=f'{name}_{h}x{w}'
        binary=scratch/(key+'.bin')
        binary.write_bytes(struct.pack('<8sIIIIf',b'AFRM0001',h,w,hp,wp,eps)+
                           struct.pack('<'+'H'*(len(bits)+len(gamma)),*(bits+gamma)))
        fixtures.append(dict(key=key,shape_hw=[h,w],binary='scratch/'+binary.name,
                             sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    (out/'fixtures.json').write_text(json.dumps(dict(passed=True,fixtures=fixtures),indent=2)+'\n')
    full=out/'fixtures.txt';pilot=out/'pilot.txt'
    full.write_text(''.join(str(out/f['binary'])+'\n' for f in fixtures))
    pilot.write_text(''.join(str(out/fixtures[i]['binary'])+'\n' for i in (1,4)))
    metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
    sim=metal.parent/'sim/libttsim_wh.so'
    revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
    digest=hashlib.sha256(sim.read_bytes()).hexdigest()
    if revision!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
        raise SystemExit('Expected pinned TT-Metal/TTSim')
    shutil.copyfile(metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml',sim.parent/'soc_descriptor.yaml')
    sources=sorted(p for p in (root/'npu/rmsnorm').rglob('*') if p.suffix in ('.cpp','.hpp'))
    sources += [root/'npu/rmsnorm/CMakeLists.txt',root/'npu/vector_add/common.hpp',
                root/'tools/report/build_rmsnorm.py',Path(__file__),root/'tools/report/audit_rmsnorm_column.py',
                root/'scripts/run_rmsnorm_column.sh']
    meta=dict(passed=False,tt_metal_revision=revision,ttsim_sha256=digest,runs=[],
              meminfo_before=Path('/proc/meminfo').read_text(),
              disk_available_bytes=shutil.disk_usage(root).free,
              source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for source in sources:
        target=out/'sources'/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();print('RESULTS='+str(out),flush=True)
    with (out/'build.log').open('w') as log:
        subprocess.run(['python3',str(root/'tools/report/build_rmsnorm.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=root/'build/rmsnorm/rmsnorm';meta['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
    def run(label,variant,cores,inputs,depth=2,repeats=1):
        command=[str(exe),str(inputs),str(out/(label+'.json')),str(depth),str(repeats),variant,str(cores)]
        start=time.perf_counter()
        with (out/(label+'.log')).open('w') as log:
            try:code=subprocess.run(command,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code=124
        meta['runs'].append(dict(label=label,command=command,exit_code=code,process_wall_seconds=time.perf_counter()-start));save()
        if code:raise SystemExit('Column RMSNorm failed: '+str(out/(label+'.log')))
        data=load(out/(label+'.json'));assert data['passed']
        print('PASS '+label,flush=True)
        return data
    pilot_data=run('pilot-column4','column',4,pilot,repeats=2)
    results={}
    for variant in ('fused','column'):
        for cores in (1,2,4):
            label=f'{variant}{cores}'
            results[label]=run(label,variant,cores,full,repeats=2 if label=='column4' else 1)
    results['column4-depth1']=run('column4-depth1','column',4,full,depth=1,repeats=2)
    baseline={c['key']:c for c in results['fused1']['cases']}
    comparisons=[]
    for label,report in results.items():
        assert len(report['cases'])==len(fixtures)
        for case in report['cases']:
            h,w=case['shape_hw'];ht,wt=(h+31)//32,(w+31)//32
            cores=case['requested_cores'];column=case['partition_axis']=='columns'
            active=min(cores,wt if column else ht);coverage=[0]*(wt if column else ht)
            assert case['active_cores']==active and case['idle_requested_cores']==cores-active
            assert case['assignment_has_no_gaps_or_overlaps'] and len(case['shards'])==active
            total=0;peak=0
            for shard in case['shards']:
                if column:
                    start,count=shard['column_tile_start'],shard['column_tile_count']
                    assert shard['tile_count']==ht*count
                    cb=(2*count+6*report['cb_depth']+3)*4096
                    assert cb==shard['cb_payload_bytes_model']
                else:
                    start,count=shard['row_group_start'],shard['row_group_count']
                    assert shard['tile_start']==start*wt and shard['tile_count']==count*wt
                    cb=(2*wt+6*report['cb_depth']+1)*4096
                assert count>0
                for unit in range(start,start+count):coverage[unit]+=1
                total+=cb;peak=max(peak,cb)
            assert coverage==[1]*len(coverage)
            assert case['cb_total_payload_bytes_model']==total and case['cb_peak_payload_bytes_model']==peak
            assert case['dram_payload_bytes_model']==3*ht*wt*4096 and case['intermediate_dram_bytes']==0
            expected_messages=2*ht*(active-1) if column else 0
            assert case['peer_data_messages_model']==expected_messages
            assert case['peer_atomic_increments_model']==expected_messages
            assert case['peer_payload_bytes_model']==128*expected_messages
            base=baseline[case['key']]
            assert case['input_fingerprint']==base['input_fingerprint'] and case['gamma_fingerprint']==base['gamma_fingerprint']
            error=max(abs(a-b) for a,b in zip(case['logical_output'],base['logical_output']))
            assert all(abs(a-b)<=1e-4+1e-4*abs(b) for a,b in zip(case['logical_output'],base['logical_output']))
            comparisons.append(dict(label=label,key=case['key'],active_cores=active,
                max_abs_difference_vs_row_single=error,bitwise_equal_to_row_single=case['output_fingerprint']==base['output_fingerprint'],
                per_core_peak_cb_bytes=peak,total_cb_bytes=total,peer_payload_bytes_model=128*expected_messages))
    summary=dict(passed=True,fixtures=len(fixtures),unique_configurations=len(comparisons),
                 total_operator_submissions=sum(c['repeats'] for data in [pilot_data,*results.values()] for c in data['cases']),
                 max_abs_error_vs_cpu_fp64=max(c['max_abs_error'] for d in results.values() for c in d['cases']),
                 max_abs_difference_vs_row_single=max(c['max_abs_difference_vs_row_single'] for c in comparisons),
                 comparisons=comparisons,performance_policy='One process per configuration; wall time includes simulator/JIT/host cost, not real chip performance')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    meta['passed']=True;save();print('COLUMN_RMSNORM_ALL_PASS '+str(out),flush=True)


if __name__=='__main__':main()
