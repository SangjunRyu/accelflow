"""Validate three-stage single-core RMSNorm on the pinned virtual Wormhole."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path

r=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path)
p.add_argument('--repeats',type=int,default=2)
a=p.parse_args()
if not 1<=a.repeats<=3: p.error('repeats must be 1..3')
out=(a.output or r/'results/rmsnorm'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
out.mkdir(parents=True,exist_ok=True)
metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
sim=Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'
revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest=hashlib.sha256(sim.read_bytes()).hexdigest()
if revision!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TT-Metal v0.79.0 / TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
if not (out/'fixtures.json').exists():
    subprocess.run(['python3',str(r/'tools/report/rmsnorm_fixtures.py'),'--output',str(out)],check=True)
sources=list((r/'npu/rmsnorm').rglob('*.cpp'))+[r/'npu/rmsnorm/CMakeLists.txt',r/'tools/report/build_rmsnorm.py',r/'tools/report/rmsnorm_fixtures.py',Path(__file__)]
metadata=dict(passed=False,tt_metal_revision=revision,ttsim_sha256=digest,runs=[],
              memory_before=Path('/proc/meminfo').read_text(),source_sha256={str(s.relative_to(r)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
for s in sources:
    dest=out/'npu-sources'/s.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,dest)
def save(): (out/'npu-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
save()
with (out/'npu-build.log').open('w') as log:
    subprocess.run(['python3',str(r/'tools/report/build_rmsnorm.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
exe=r/'build/rmsnorm/rmsnorm'
metadata['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
for label,depth,repeats,fixtures in [('pilot',1,1,'pilot-fixtures.txt'),('depth1',1,a.repeats,'npu-fixtures.txt'),('depth2',2,a.repeats,'npu-fixtures.txt')]:
    command=[str(exe),str(out/fixtures),str(out/f'npu-{label}.json'),str(depth),str(repeats)]
    started=time.perf_counter()
    with (out/f'npu-{label}.log').open('w') as log:
        try:
            code=subprocess.run(command,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        except subprocess.TimeoutExpired: code=124
    metadata['runs'].append(dict(label=label,command=command,exit_code=code,wall_seconds=time.perf_counter()-started));save()
    if code: raise SystemExit('NPU RMSNorm failed: '+str(out/f'npu-{label}.log'))
    data=json.loads((out/f'npu-{label}.json').read_text());assert data['passed']
    print('NPU_RMSNORM_PASS '+label+' cases='+str(len(data['cases'])),flush=True)
one=json.loads((out/'npu-depth1.json').read_text());two=json.loads((out/'npu-depth2.json').read_text())
assert len(one['cases'])==len(two['cases'])==20
for x,y in zip(one['cases'],two['cases'],strict=True):
    assert (x['key'],x['input_fingerprint'],x['gamma_fingerprint'],x['output_fingerprint'])==(y['key'],y['input_fingerprint'],y['gamma_fingerprint'],y['output_fingerprint'])
summary=dict(passed=True,configurations=40,operator_submissions=40*a.repeats,pilot_operator_submissions=2,
             stages_per_operator=3,core_count=1,max_abs_error=max(x['max_abs_error'] for x in one['cases']),
             depth_fingerprints_equal=True,real_npu_performance_measured=False)
(out/'npu-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
metadata['passed']=True;save()
print('NPU_RMSNORM_ALL_PASS '+str(out),flush=True)
