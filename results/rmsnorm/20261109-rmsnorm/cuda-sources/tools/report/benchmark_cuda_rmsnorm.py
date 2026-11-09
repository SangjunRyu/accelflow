"""Validate unfused/fused RMSNorm on common fixtures and retain every timing sample."""
import argparse
from datetime import datetime,timezone
import hashlib,json,random,statistics,subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--executable',type=Path,required=True)
p.add_argument('--local-fixtures',type=Path,required=True)
p.add_argument('--trials',type=int,default=3)
a=p.parse_args()
if not 1<=a.trials<=10:p.error('trials must be 1..10')
out=a.output
manifest=json.loads((out/'fixtures.json').read_text(encoding='utf-8-sig'));assert manifest['passed']
jobs=[(f,v,t) for f in manifest['fixtures'] for v in ['unfused','fused'] for t in range(1,(a.trials if f['performance'] else 1)+1)]
random.Random(2026).shuffle(jobs)
measurements=[]
for fixture,variant,trial in jobs:
    binary=a.local_fixtures/Path(fixture['binary']).name
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==fixture['sha256']
    key=fixture['key']+'_'+variant+f'_trial{trial}'
    report=out/(key+'.json')
    samples,batch=(15,20) if fixture['performance'] else (5,1)
    command=[str(a.executable),str(binary),str(report),variant,str(samples),str(batch)]
    with (out/(key+'.log')).open('w',encoding='utf-8') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
    if code:raise SystemExit('CUDA RMSNorm failed: '+key)
    data=json.loads(report.read_text());assert data['passed'] and data['validated_outputs']==1+2*samples
    measurements.append(dict(key=fixture['key'],variant=variant,trial=trial,performance=fixture['performance'],
                             report=report.name,command=command,exit_code=code,max_abs_error=data['max_abs_error'],
                             validated_outputs=data['validated_outputs'],
                             operator_ms=statistics.median(data['operator_event_batch_average_ms']),
                             pipeline_ms=statistics.median(data['host_pipeline_ms'])))
    (out/'cuda-measurements.json').write_text(json.dumps(measurements,indent=2)+'\n',encoding='utf-8')
    print('CUDA_RMSNORM_PASS '+key,flush=True)
rows=[]
for fixture in manifest['fixtures']:
    if not fixture['performance']:continue
    variants={}
    for v in ['unfused','fused']:
        runs=[x for x in measurements if x['key']==fixture['key'] and x['variant']==v]
        variants[v]=dict(operator_ms=statistics.median(x['operator_ms'] for x in runs),
                         pipeline_ms=statistics.median(x['pipeline_ms'] for x in runs),
                         process_medians_ms=[x['operator_ms'] for x in runs])
    rows.append(dict(key=fixture['key'],shape_hw=fixture['shape_hw'],variants=variants,
                     observed_speedup=variants['unfused']['operator_ms']/variants['fused']['operator_ms']))
summary=dict(passed=True,timestamp_utc=datetime.now(timezone.utc).isoformat(),processes=len(measurements),
             validated_outputs=sum(x['validated_outputs'] for x in measurements),
             max_abs_error=max(x['max_abs_error'] for x in measurements),rows=rows,
             executable_sha256=hashlib.sha256(a.executable.read_bytes()).hexdigest(),
             measurement='CUDA Event batch-average operator interval; 3 launches versus 1; scheduling and host submission gaps may be included',
             bank_conflict_counters=False)
(out/'cuda-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print('CUDA_RMSNORM_ALL_PASS '+str(out),flush=True)
