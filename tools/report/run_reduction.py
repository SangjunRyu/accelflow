"""Build and validate single-core BF16-input / FP32-output row-sum on virtual Wormhole."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,platform,shutil,subprocess,time
from pathlib import Path

r=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repeats',type=int,default=3)
args=parser.parse_args()
if not 1<=args.repeats<=5:raise SystemExit('repeats must be 1..5')
metal=Path(os.environ.get('TT_METAL_HOME',str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim=Path(os.environ.get('TT_METAL_SIMULATOR',str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest=hashlib.sha256(sim.read_bytes()).hexdigest()
if revision!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
 raise SystemExit('Expected pinned TT-Metal v0.79.0 and TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
out=r/'results/reduction'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
print(out,flush=True)
sources=list((r/'npu/reduction').rglob('*.cpp'))+[r/'npu/reduction/CMakeLists.txt',r/'npu/vector_add/common.hpp',r/'tools/report/build_reduction.py',Path(__file__),r/'scripts/run_reduction.sh']
meta=dict(passed=False,operator='row_sum',scope='single core, last dimension only; no hardware performance claim',
 tt_metal_revision=revision,ttsim_sha256=digest,host=platform.platform(),
 compiler=subprocess.check_output(['clang++-20','--version'],text=True),meminfo_before=Path('/proc/meminfo').read_text(),
 source_sha256={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},runs=[])
for source in sources:
 dest=out/'sources'/source.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
def save(): (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
save()
with (out/'build.log').open('w') as log:
 code=subprocess.run(['python3',str(r/'tools/report/build_reduction.py')],cwd=r,stdout=log,stderr=subprocess.STDOUT).returncode
meta['build_exit_code']=code;save()
if code:raise SystemExit('Host build failed: '+str(out/'build.log'))
binary=r/'build/reduction/reduction'
meta['executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
def execute(label,depth,repeats,selector=None):
 report=out/(label+'.json');command=[str(binary),str(report),str(depth),str(repeats)]
 if selector:command.append(selector)
 with (out/(label+'.log')).open('w') as log:
  log.write('Command: '+repr(command)+'\n');log.flush()
  started=time.perf_counter()
  try:code=subprocess.run(command,cwd=metal,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
  except subprocess.TimeoutExpired:code=124
  log.write(f'\nexample_exit_code={code}\n')
 meta['runs'].append(dict(label=label,command=command,exit_code=code,process_wall_seconds=time.perf_counter()-started));save()
 if code:raise SystemExit('Reduction failed: '+str(out/(label+'.log')))
 data=json.loads(report.read_text());assert data['passed']
 print('PASS '+label+' cases='+str(len(data['cases'])),flush=True)
 return data

# Fail early on output layout/precision before running the full fixture list.
execute('pilot_small',1,1,'deterministic:2,3')
execute('pilot_fp32_257',1,1,'ones:2,257')
results=[execute('depth'+str(depth),depth,args.repeats) for depth in (1,2)]
observed={}
for data in results:
 for row in data['cases']:
  h,w=row['shape_hw'];ht,wt=row['tile_grid_hw'];key=(row['name'],h,w)
  assert ht==(h+31)//32 and wt==(w+31)//32
  assert row['passed'] and row['all_submission_outputs_equal'] and row['output_padded_lanes_zero'] and row['entire_final_prefix_tile_verified']
  assert row['cb_capacity_bytes_model']==data['cb_depth']*6144
  assert row['cumsum_tile_calls_model']==ht*wt and row['input_tile_reads_model']==ht*wt
  assert row['dram_read_payload_bytes_model']==ht*wt*2048
  assert row['dram_write_payload_bytes_model']==ht*4096
  if row['name']!='fractional':assert row['max_abs_error']==0
  fp=(row['input_fingerprint'],row['output_fingerprint'])
  if key in observed:assert observed[key]==fp,('depth changed input/output',key)
  observed[key]=fp
  if row['name']=='ones':assert row['row_sums']==[float(w)]*h
  if row['name']=='large_exact':assert row['row_sums']==[8200.0]*h
summary=dict(passed=True,full_suite_cases=sum(len(x['cases']) for x in results),
 full_suite_submissions=sum(len(x['cases'])*args.repeats for x in results),pilot_submissions=2,
 shapes_and_inputs_matched_across_depth=True,
 cb_bytes_per_core_by_depth={str(x['cb_depth']):x['cases'][0]['cb_capacity_bytes_model'] for x in results},
 max_abs_error=max(row['max_abs_error'] for data in results for row in data['cases']),
 fractional_max_abs_error=max(row['max_abs_error'] for data in results for row in data['cases'] if row['name']=='fractional'),
 cases=results[0]['cases'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
meta['passed']=True;meta['meminfo_after']=Path('/proc/meminfo').read_text();save()
print('REDUCTION_VALIDATION_PASS '+str(summary['full_suite_cases'])+' cases / '+str(summary['full_suite_submissions'])+' submissions; '+str(out),flush=True)
