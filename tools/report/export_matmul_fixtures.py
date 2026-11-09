"""Export exact TT-Metal BF16 inputs and validate the NPU producer before CUDA use."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess

r=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path)
args=p.parse_args()
out=args.output or r/'results/cuda_matmul'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True,exist_ok=True)
metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
sim=Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'
commit=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest=hashlib.sha256(sim.read_bytes()).hexdigest()
if commit!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
 raise SystemExit('Expected pinned TT-Metal v0.79.0 and TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
with (out/'npu-build.log').open('w') as log:
 subprocess.run(['python3',str(r/'tools/report/build_matmul.py')],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
fixtures=[('deterministic',(2,3,2)),('zero',(33,35,31)),('identity',(32,32,32)),
 ('signed_random',(31,33,35)),('fractional',(33,257,65)),('signed_random',(65,257,97)),
 ('signed_random',(128,128,128)),('signed_random',(320,320,320)),('signed_random',(640,320,640))]
meta=dict(passed=False,tt_metal_revision=commit,ttsim_sha256=digest,fixtures=[],
 source_sha256={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),r/'npu/matmul/matmul.cpp',r/'npu/matmul/benchmark.hpp']})
for name,shape in fixtures:
 key=name+'_'+'x'.join(map(str,shape)); binary=out/'scratch'/f'{key}.bin'; report=out/f'npu_{key}.json'
 command=[str(r/'build/matmul/matmul'),str(report),'multicore','2','4','--benchmark','5',name+':'+','.join(map(str,shape)),'--export-inputs',str(binary)]
 with (out/f'npu_{key}.log').open('w') as log:
  proc=subprocess.run(command,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 if proc.returncode:raise SystemExit('NPU fixture producer failed: '+key)
 data=json.loads(report.read_text()); assert data['passed'] and len(data['cases'])==1
 meta['fixtures'].append(dict(key=key,name=name,shape=shape,performance=shape in [(128,128,128),(320,320,320),(640,320,640)],
  binary=str(binary.relative_to(out)),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),npu_report=report.name,command=command,exit_code=proc.returncode))
 print('NPU_FIXTURE_PASS '+key,flush=True)
 (out/'fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
meta['passed']=True
(out/'fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
print('FIXTURES_ALL_PASS '+str(out),flush=True)
