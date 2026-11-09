"""Run both real simulator variants and retain correctness/structural evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
metal = Path(os.environ.get('TT_METAL_HOME', str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim = Path(os.environ.get('TT_METAL_SIMULATOR', str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
expected = '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a'
if hashlib.sha256(sim.read_bytes()).hexdigest() != expected:
    raise SystemExit('Expected the pinned TTSim v1.11.1 Wormhole binary')
descriptor = metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml'
if not (sim.parent/'soc_descriptor.yaml').is_file() or \
   (sim.parent/'soc_descriptor.yaml').read_bytes() != descriptor.read_bytes():
    (sim.parent/'soc_descriptor.yaml').write_bytes(descriptor.read_bytes())
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out = root/'results/vector_add'/stamp
out.mkdir(parents=True)
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1')
def capture(*args):
    return subprocess.check_output(args, text=True).strip()
metadata = {
    'date_utc': stamp, 'backend':'ttsim', 'architecture':'wormhole', 'operator':'vector_add',
    'tt_metal_commit':capture('git','-C',str(metal),'rev-parse','HEAD'), 'tt_metal_tag':'v0.79.0',
    'ttsim_tag':'v1.11.1', 'ttsim_sha256':expected, 'project_git_commit':None,
    'compiler':capture('clang++-20','--version'), 'cmake':capture('cmake','--version'),
    'python':sys.version, 'os_release':Path('/etc/os-release').read_text(), 'kernel':capture('uname','-r'),
    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root/'npu/vector_add').rglob('*')) if p.is_file()},
    'measurement_policy':'Functional/structural evidence only; no simulator latency comparison', 'runs':[]
}
success = True
try:
    for variant, executable, extra in (('riscv','riscv',[]), ('tiled_depth1','tiled',['1']), ('tiled','tiled',['2'])):
        binary = root/'build/vector_add'/f'vector_add_{executable}'
        report = out/f'{variant}.json'
        with (out/f'{variant}.log').open('w') as log:
            command = [str(binary),str(report),*extra]
            log.write('Command: '+repr(command)+'\n'); log.flush()
            try:
                run = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                code = run.returncode
            except subprocess.TimeoutExpired:
                code = 124
            log.write(f'example_exit_code={code}\n')
        passed = code == 0 and report.is_file() and json.loads(report.read_text()).get('passed') is True
        metadata['runs'].append({'variant':variant,'exit_code':code,'passed':passed,'report':report.name})
        print((out/f'{variant}.log').read_text()[-6000:], flush=True)
        success &= passed
        if not passed: break
finally:
    metadata['passed'] = success and len(metadata['runs']) == 3
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
raise SystemExit(0 if metadata['passed'] else 1)
