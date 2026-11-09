"""Run the Windows CUDA executable and compare validated input identities with TT-Metal."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,platform,random,statistics,subprocess
from pathlib import Path

r=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--executable',type=Path,required=True)
p.add_argument('--local-fixtures',type=Path,required=True)
p.add_argument('--trials',type=int,default=3)
p.add_argument('--variants',nargs='+',choices=['naive','tiled','texture_tiled'],default=['naive','tiled'])
args=p.parse_args()
if args.trials<1 or args.trials>5:raise SystemExit('trials 1..5')
out=args.output
fixtures=json.loads((out/'fixtures.json').read_text());assert fixtures['passed']
meta=dict(passed=False,host=platform.platform(),python=platform.python_version(),backend='Windows native CUDA',
 timestamp_utc=datetime.now(timezone.utc).isoformat(),trials=args.trials,variants=args.variants,runs=[],
 executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
 source_sha256={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'cuda/matmul/matmul.cu',Path(__file__),r/'scripts/run_cuda_matmul.ps1']})
records=[]
def save():
 (out/'cuda-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
save()
jobs=[]
for fixture in fixtures['fixtures']:
 for variant in args.variants:
  for trial in range(args.trials if fixture['performance'] else 1):jobs.append((fixture,variant,trial+1))
random.Random(20261109).shuffle(jobs)
for fixture,variant,trial in jobs:
 name=f"cuda_{fixture['key']}_{variant}_trial{trial}"
 report=out/(name+'.json'); binary=args.local_fixtures/Path(fixture['binary']).name
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==fixture['sha256']
 command=[str(args.executable),str(binary),str(report),variant,'20','15','20']
 with (out/(name+'.log')).open('w',encoding='utf-8') as log:
  proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 meta['runs'].append(dict(name=name,command=command,exit_code=proc.returncode));save()
 if proc.returncode:raise SystemExit('CUDA run failed: '+str(out/(name+'.log')))
 row=json.loads(report.read_text());assert row['passed'] and row['validated_sample_outputs']==46
 npu=json.loads((out/fixture['npu_report']).read_text())['cases'][0]
 for key in ('input_fingerprint_a','input_fingerprint_b'):assert row[key]==npu[key],(name,key)
 same_output=row['bf16_output_fingerprint']==npu['output_fingerprint']
 if fixture['name']!='fractional':assert same_output,(name,'BF16 output mismatch')
 for key in ('kernel_event_batch_average_ms','host_resident_launch_finish_ms','h2d_pinned_event_ms','d2h_pinned_event_ms','host_pipeline_h2d_kernel_d2h_ms'):
  assert len(row[key])==15 and all(x>0 for x in row[key]),(name,key)
 records.append(dict(fixture=fixture['key'],performance=fixture['performance'],variant=variant,trial=trial,
  input_fingerprints_match_npu=True,bf16_output_fingerprint_matches_npu=same_output,report=report.name,
  medians_ms={key:statistics.median(row[key]) for key in ('kernel_event_batch_average_ms','host_resident_launch_finish_ms','h2d_pinned_event_ms','d2h_pinned_event_ms','host_pipeline_h2d_kernel_d2h_ms')},
  resource_creation_host_ms=row['resource_creation_host_ms'],max_abs_error=row['max_abs_error'],bf16_rounded_max_abs_error=row['bf16_rounded_max_abs_error']))
 (out/'cuda-measurements.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
 print('PASS '+name,flush=True)
summary=[]
for fixture in fixtures['fixtures']:
 if not fixture['performance']:continue
 for variant in args.variants:
  rows=[x for x in records if x['fixture']==fixture['key'] and x['variant']==variant]
  keys=rows[0]['medians_ms'].keys()
  summary.append(dict(fixture=fixture['key'],shape=fixture['shape'],variant=variant,
   medians_ms={key:statistics.median(x['medians_ms'][key] for x in rows) for key in keys},
   trial_kernel_medians_ms=[x['medians_ms']['kernel_event_batch_average_ms'] for x in sorted(rows,key=lambda x:x['trial'])],
   resource_creation_host_median_ms=statistics.median(x['resource_creation_host_ms'] for x in rows),
   all_bf16_outputs_match_npu=all(x['bf16_output_fingerprint_matches_npu'] for x in rows)))
(out/'cuda-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
meta['passed']=True;meta['validated_outputs']=sum(json.loads((out/(x['name']+'.json')).read_text())['validated_sample_outputs'] for x in meta['runs']);save()
for row in summary:print(row['shape'],row['variant'],row['medians_ms'],flush=True)
print('CUDA_ALL_PASS '+str(out),flush=True)
