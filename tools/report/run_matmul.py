"""Execute same-input baseline/reuse/blocked and CB-depth experiments on virtual Wormhole."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
metal=Path(os.environ.get('TT_METAL_HOME',str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim=Path(os.environ.get('TT_METAL_SIMULATOR',str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
digest=hashlib.sha256(sim.read_bytes()).hexdigest()
if digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TTSim v1.11.1 Wormhole')
revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
if revision!='de546d3b146758714d900f11b218c8f9c805f410': raise SystemExit('Unexpected TT-Metal revision')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
         TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
out=root/'results/matmul'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
sources=[root/'tools/report/run_matmul.py',root/'tools/report/assess_matmul.py',root/'tools/report/build_matmul.py',root/'scripts/run_matmul.sh']+[p for p in (root/'npu/matmul').rglob('*') if p.is_file()]+[root/'npu/vector_add/common.hpp']
metadata={'backend':'ttsim','architecture':'wormhole','operator':'matmul','tt_metal_tag':'v0.79.0',
    'tt_metal_commit':revision,'ttsim_tag':'v1.11.1','ttsim_sha256':digest,'project_git_commit':None,
    'date_utc':datetime.now(timezone.utc).isoformat(),'os_release':Path('/etc/os-release').read_text(),
    'kernel':subprocess.check_output(['uname','-r'],text=True).strip(),'python':sys.version,
    'compiler':subprocess.check_output(['clang++-20','--version'],text=True).strip(),
    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'runtime_root':str(metal),'runs':[],'measurement_policy':'No simulator timing used as NPU performance'}
ok=True
try:
    for variant,depth,chunk in ([(v,d,0) for v in ('baseline','reuse_a','blocked') for d in (1,2)] +
                                [('streamed',d,c) for c in (1,2) for d in (1,2)]):
        name=f'{variant}_depth{depth}' + (f'_chunk{chunk}' if chunk else '')
        report=out/f'{name}.json'
        command=[str(root/'build/matmul/matmul'),str(report),variant,str(depth)]
        if chunk: command.append(str(chunk))
        with (out/f'{name}.log').open('w') as log:
            log.write('Command: '+repr(command)+'\n'); log.flush()
            try:
                code=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
            except subprocess.TimeoutExpired:
                code=124
            log.write(f'example_exit_code={code}\n')
        passed=code==0 and report.is_file() and json.loads(report.read_text()).get('passed') is True
        metadata['runs'].append({'variant':variant,'depth':depth,'chunk':chunk,'exit_code':code,'passed':passed,'report':report.name})
        print((out/f'{name}.log').read_text()[-6500:],flush=True)
        ok &= passed
        if not passed: break
finally:
    metadata['passed']=ok and len(metadata['runs'])==10
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
raise SystemExit(0 if metadata['passed'] else 1)
