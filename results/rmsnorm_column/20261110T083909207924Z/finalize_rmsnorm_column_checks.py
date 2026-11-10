"""Run matched-repeat timing control and legacy unfused regression after the suite."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root=Path('/home/ysjun5656/dev/accelflow')
out=Path(sys.argv[1]).resolve()
metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
exe=root/'build/rmsnorm/rmsnorm'
meta=json.loads((out/'metadata.json').read_text())
assert meta['passed'] and hashlib.sha256(exe.read_bytes()).hexdigest()==meta['executable_sha256']
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
         TT_METAL_SIMULATOR=str(metal.parent/'sim/libttsim_wh.so'),TT_METAL_SLOW_DISPATCH_MODE='1')
cases=json.loads((out/'fixtures.json').read_text())['fixtures']
legacy=out/'legacy-fixtures.txt'
legacy.write_text(''.join(str(out/f['binary'])+'\n' for f in cases[:2]))
runs=[]
for label,variant,cores,inputs in [('column4-once','column',4,out/'fixtures.txt'),
                                    ('legacy-unfused','unfused',1,legacy)]:
    command=[str(exe),str(inputs),str(out/(label+'.json')),'2','1',variant,str(cores)]
    start=time.perf_counter()
    with (out/(label+'.log')).open('w') as log:
        process=subprocess.run(command,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    record=dict(label=label,command=command,exit_code=process.returncode,process_wall_seconds=time.perf_counter()-start)
    runs.append(record)
    assert process.returncode==0,label
    data=json.loads((out/(label+'.json')).read_text());assert data['passed']
    reference=json.loads((out/('column4.json' if variant=='column' else 'fused1.json')).read_text())
    bykey={c['key']:c for c in reference['cases']}
    for case in data['cases']:
        base=bykey[case['key']]
        assert case['input_fingerprint']==base['input_fingerprint']
        assert case['gamma_fingerprint']==base['gamma_fingerprint']
        assert case['output_fingerprint']==base['output_fingerprint']
    print('FOLLOWUP_PASS '+label,flush=True)
result=dict(passed=True,runs=runs,additional_operator_submissions=9,
    main_suite_operator_submissions=67,total_operator_submissions=76,
    matched_timing_control='column4-once matches other depth2 full-suite processes: seven cases, one submission each; one process each, JIT/order uncontrolled',
    legacy_scope='Two fixtures, unfused three-stage output bitwise equal to row-fused baseline; not a rerun of all historical experiments',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'followup-checks.json').write_text(json.dumps(result,indent=2)+'\n')
(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print('FOLLOWUP_ALL_PASS',flush=True)
