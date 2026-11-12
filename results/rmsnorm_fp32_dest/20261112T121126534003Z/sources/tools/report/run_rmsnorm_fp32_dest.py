"""Test native and locally adapted FP32-to-Dest transport, then no-split RMSNorm."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import time


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=(args.output or root/'results/rmsnorm_fp32_dest'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if (out/'metadata.json').exists():raise SystemExit('Use a new result directory')
    out.mkdir(parents=True,exist_ok=True)
    metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
    sim=metal.parent/'sim/libttsim_wh.so'
    revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
    digest=hashlib.sha256(sim.read_bytes()).hexdigest()
    if revision!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
        raise SystemExit('Expected original pinned TT-Metal/TTSim')
    sources=[p for p in (root/'npu/rmsnorm_fp32_dest').rglob('*') if p.suffix in ('.cpp','.hpp','.txt')]
    sources += [root/'npu/vector_add/common.hpp',Path(__file__),root/'tools/report/build_rmsnorm_fp32_dest.py',
                root/'tools/report/rmsnorm_fixtures.py',root/'tools/report/audit_rmsnorm_fp32_dest.py',
                root/'scripts/run_rmsnorm_fp32_dest.sh']
    meta=dict(passed=False,runs=[],tt_metal_revision=revision,ttsim_sha256=digest,
              meminfo_before=Path('/proc/meminfo').read_text(),
              source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for p in sources:
        target=out/'sources'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();print('RESULTS='+str(out),flush=True)
    with (out/'build.log').open('w') as log:
        subprocess.run(['python3',str(root/'tools/report/build_rmsnorm_fp32_dest.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=root/'build/rmsnorm_fp32_dest/rmsnorm_fp32_dest'
    meta['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
             TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
    shutil.copyfile(metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml',sim.parent/'soc_descriptor.yaml')
    def run(label,arguments,expected=0):
        command=[str(exe),*arguments];start=time.perf_counter()
        with (out/(label+'.log')).open('w') as log:
            try:code=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code=124
        meta['runs'].append(dict(label=label,command=command,exit_code=code,expected_exit_code=expected,
                                 process_wall_seconds=time.perf_counter()-start));save()
        if code!=expected:raise SystemExit('Unexpected result: '+str(out/(label+'.log')))
        print('CHECK '+label+' exit='+str(code),flush=True)
        return load(out/(label+'.json')) if (out/(label+'.json')).exists() else None
    transport=[]
    for op in ('copy','transpose'):
        for mode in ('default','direct','compat'):
            label=f'probe-{mode}-{op}'
            data=run(label,['probe',str(out/(label+'.json')),mode,op],expected=1 if mode=='direct' else 0)
            if mode=='direct':
                assert 'unpack_to_dst=0' in (out/(label+'.log')).read_text()
            else:
                assert data['passed'] and data['bit_mismatches']==(1024 if mode=='default' else 0)
            transport.append(dict(mode=mode,operation=op,native_rejected=mode=='direct',
                                  bit_mismatches=None if data is None else data['bit_mismatches']))
    subprocess.run(['python3',str(root/'tools/report/rmsnorm_fixtures.py'),'--output',str(out)],check=True)
    original_manifest=load(out/'fixtures.json')
    fixtures=[dict(f,fixture_dtype='BF16') for f in original_manifest['fixtures'] if not f['performance']]
    shapes=[('fp32_lowbits',3,33,1e-5),('fp32_fractional',33,65,1e-5),('fp32_wide',1,1025,1e-5),
            ('fp32_tail',65,129,1e-5),('fp32_tiny',3,31,1e-5),('fp32_zero',1,1,1e-5),
            ('fp32_gamma',17,257,1e-5),('fp32_epsilon',2,33,1e-3),('fp32_offset',1,257,1e-6)]
    for name,h,w,eps in shapes:
        hp,wp=(h+31)//32*32,(w+31)//32*32;rng=random.Random(20261112+h*4099+w)
        values=[]
        for row in range(hp):
            for col in range(wp):
                value=rng.uniform(-2,2)
                if name=='fp32_lowbits':value=(-1 if col%3 else 1)*(1+((row*wp+col)%4096+1)*2**-20)
                if name in ('fp32_tiny','fp32_epsilon'):value=rng.uniform(-0.0002,0.0002)
                if name=='fp32_offset':value=128+((col*13)%1024+1)*2**-12
                if name=='fp32_zero' or row>=h or col>=w:value=0.0
                values.append(value)
        gamma=[rng.uniform(-1.5,1.5) if col<w else 0.0 for col in range(wp)]
        if name=='fp32_gamma':gamma=[0.0 if col%7==0 or col>=w else gamma[col] for col in range(wp)]
        key=f'{name}_{h}x{w}_eps{eps:g}';binary=out/'scratch'/(key+'.bin')
        binary.write_bytes(struct.pack('<8sIIIIf',b'AFRF0001',h,w,hp,wp,eps)+
                           struct.pack('<'+'f'*(len(values)+len(gamma)),*(values+gamma)))
        fixtures.append(dict(key=key,shape_hw=[h,w],binary='scratch/'+binary.name,
                             sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),fixture_dtype='FP32',performance=False))
    (out/'fixtures.json').write_text(json.dumps(dict(passed=True,fixtures=fixtures),indent=2)+'\n')
    full=out/'all-fixtures.txt';pilot=out/'fp32-pilot.txt'
    full.write_text(''.join(str(out/f['binary'])+'\n' for f in fixtures))
    pilot.write_text(''.join(str(out/f['binary'])+'\n' for f in (fixtures[0],fixtures[9],fixtures[20],fixtures[21])))
    run('rms-pilot',['run',str(pilot),str(out/'rms-pilot.json'),'1','1','compat'])
    results={d:run(f'compat-depth{d}',['run',str(full),str(out/f'compat-depth{d}.json'),str(d),'2','compat']) for d in (1,2)}
    comparisons=[]
    for depth,data in results.items():
        assert data['passed'] and data['residual_bit_splitting'] is False and data['unpack_to_dest_fp32']
        old=load(root/f'results/rmsnorm_fused/20261110T051656706592Z/fused-depth{depth}.json')
        old_cases={c['key']:c for c in old['cases']}
        for case in data['cases']:
            assert case['passed'] and case['repeats_bitwise_equal']
            if case['fixture_dtype'].startswith('BF16'):
                base=old_cases[case['key']]
                assert case['input_fingerprint']==base['input_fingerprint'] and case['gamma_fingerprint']==base['gamma_fingerprint']
                difference=max(abs(a-b) for a,b in zip(case['logical_output'],base['logical_output']))
                assert all(abs(a-b)<=1e-4+1e-4*abs(b) for a,b in zip(case['logical_output'],base['logical_output']))
                comparisons.append(dict(key=case['key'],depth=depth,max_abs_difference=difference))
    a={c['key']:c for c in results[1]['cases']}
    assert all(c['output_fingerprint']==a[c['key']]['output_fingerprint'] for c in results[2]['cases'])
    summary=dict(passed=True,native_direct_transport_passed=False,compat_transport_bitwise_exact=True,
        transport=transport,original_bf16_fixtures=20,actual_fp32_fixtures=9,configurations=58,
        rmsnorm_evaluations=120,program_submissions=360,transport_submissions=6,
        residual_bit_splitting=False,fp32_dest_acc_en=True,marked_input_cbs=[0,1,2],
        max_abs_error=max(c['max_abs_error'] for data in results.values() for c in data['cases']),
        max_sum_square_error=max(c['max_sum_square_error'] for data in results.values() for c in data['cases']),
        square_bit_differences=sum(c['square_bit_differences'] for data in results.values() for c in data['cases']),
        bf16_comparisons=comparisons,depth_outputs_bitwise_equal=True,
        limitation='Single core, 3-stage precision experiment with local TEN-3868 compatibility adaptation; not native API-only success or chip performance')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');meta['passed']=True;save()
    print('FP32_DEST_RMSNORM_ALL_PASS '+str(out),flush=True)


if __name__=='__main__':main()
