"""Validate actual A-multicast transport, numerical equivalence and repeated handshakes."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
metal=Path(os.environ.get('TT_METAL_HOME',str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim=Path(os.environ.get('TT_METAL_SIMULATOR',str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest=hashlib.sha256(sim.read_bytes()).hexdigest()
if revision!='de546d3b146758714d900f11b218c8f9c805f410' or digest!='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TT-Metal v0.79.0 / TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
         TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
out=root/'results/matmul_mcast'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
sources=[p for p in (root/'npu/matmul').rglob('*') if p.is_file()]
sources += [Path(__file__),root/'scripts/run_mcast_matmul.sh',root/'npu/vector_add/common.hpp']
metadata={'date_utc':datetime.now(timezone.utc).isoformat(),'backend':'ttsim',
          'tt_metal_revision':revision,'ttsim_sha256':digest,'runs':[],'repeat_runs':[],'passed':False,
          'compiler':subprocess.check_output(['clang++-20','--version'],text=True),
          'kernel':subprocess.check_output(['uname','-r'],text=True),
          'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}

def execute(name,variant,depth,cores,selector=None):
    path=out/f'{name}.json'
    command=[str(root/'build/matmul/matmul'),str(path),variant,str(depth)]
    if variant in ('mcast','multicore'):
        command.append(str(cores))
    if selector:
        command += ['--benchmark','5',selector]
    with (out/f'{name}.log').open('w') as log:
        log.write('Command: '+repr(command)+'\n');log.flush()
        try:
            code=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        except subprocess.TimeoutExpired:
            code=124
        log.write(f'example_exit_code={code}\n')
    record={'name':name,'variant':variant,'depth':depth,'cores':cores,'exit_code':code}
    metadata['repeat_runs' if selector else 'runs'].append(record)
    if code!=0 or not path.is_file():
        raise RuntimeError(f'Failed {name}: exit {code}; see {out/name}.log')
    report=json.loads(path.read_text())
    if not report['passed']:
        raise RuntimeError(f'Failed correctness: {name}')
    print(f'PASS {name}: {len(report["cases"])} fixtures',flush=True)
    return report

try:
    reports={}
    # 4코어 multicast와 tail부터 검사한 뒤 대조군을 실행합니다.
    configs=[('mcast',d,c) for c in (4,2,1) for d in (1,2)]
    configs += [('baseline',d,1) for d in (1,2)]+[('multicore',d,4) for d in (1,2)]
    for variant,depth,cores in configs:
        name=f'{variant}_cores{cores}_depth{depth}'
        reports[name]=execute(name,variant,depth,cores)
        if len(reports[name]['cases'])!=20:
            raise RuntimeError('Expected 20 common fixtures')

    baseline=reports['baseline_cores1_depth2']['cases']
    comparisons=[]
    coverage={'tail_receiver':False,'multi_row':False,'CB_wrap':False,
              'three_active_cores':False,'four_active_cores':False,'single_core_fallback':False,'large_K':False}
    for index,base in enumerate(baseline):
        mt,kt,nt=base['tile_grid_mkn']
        total=mt*kt*nt
        modes={}
        for name,report in reports.items():
            row=report['cases'][index]
            for key in ('name','shape_mkn','input_fingerprint_a','input_fingerprint_b','output_fingerprint'):
                if row[key]!=base[key]:
                    raise RuntimeError(f'Controlled comparison mismatch: {name}/{key}')
            if not all(row[k] for k in ('passed','layout_matches_official_tilizer','output_sentinel_overwritten','padding_verified','fp32_destination_accumulation')):
                raise RuntimeError('Functional contract failed')
            if report['variant']!='mcast':
                continue
            p=min(report['requested_cores'],nt)
            groups=(nt+p-1)//p
            reads=mt*kt*groups
            dist=row['distribution']
            expected={'active_cores':p,'a_dram_reads_model':reads,
                      'a_multicast_calls_model':reads if p>1 else 0,
                      'a_receiver_deliveries_model':reads*(p-1),
                      'unused_tail_a_tiles_model':mt*kt*(groups*p-nt)}
            for key,value in expected.items():
                if dist[key]!=value:
                    raise RuntimeError(f'Incorrect distribution model: {key}')
            seen=[]
            physical=[]
            for rank,assignment in enumerate(dist['assignments']):
                if assignment['logical_core']!=[rank,0]:
                    raise RuntimeError('Unexpected logical mapping')
                expected_ids=[m*nt+n for m in range(mt) for n in range(rank,nt,p)]
                if assignment['output_tile_ids']!=expected_ids:
                    raise RuntimeError('Column partition mismatch')
                seen+=assignment['output_tile_ids']
                physical.append(assignment['physical_core'])
            if sorted(seen)!=list(range(mt*nt)):
                raise RuntimeError('Duplicate or missing output tile')
            if any(y!=physical[0][1] or x!=physical[0][0]+rank for rank,(x,y) in enumerate(physical)):
                raise RuntimeError('Multicast rectangle is not exactly the receiver core row')
            depth=report['stream_cb_depth']
            expected_row={'core_count':p,'a_tile_reads_model':reads,'b_tile_reads_model':total,
                          'dram_read_payload_model':2048*(reads+total),'dram_write_payload_model':2048*mt*nt,
                          'matmul_tile_calls_model':total,'read_barriers_model':reads+total,
                          'cb_capacity_bytes_model':3*depth*2048,'cb_capacity_bytes_total_model':p*3*depth*2048}
            for key,value in expected_row.items():
                if row[key]!=value:
                    raise RuntimeError(f'Incorrect source model: {key}')
            modes[name]={'active_cores':p,'A_reads':reads,'B_reads':total,
                         'read_payload_bytes_model':row['dram_read_payload_model'],
                         'reduction_vs_baseline_model':1-row['dram_read_payload_model']/base['dram_read_payload_model'],
                         'tail_a_discarded_model':dist['unused_tail_a_tiles_model']}
            coverage['tail_receiver'] |= p>1 and nt%p!=0
            coverage['multi_row'] |= p>1 and mt>1
            coverage['CB_wrap'] |= mt*kt*groups>depth
            coverage['three_active_cores'] |= p==3
            coverage['four_active_cores'] |= p==4
            coverage['single_core_fallback'] |= p==1
            coverage['large_K'] |= p>1 and kt==16
        comparisons.append({'name':base['name'],'shape_mkn':base['shape_mkn'],'controlled_outputs_match':True,'mcast_models':modes})
    if not all(coverage.values()):
        raise RuntimeError(f'Incomplete coverage: {coverage}')

    # 같은 program을 재제출해 semaphore와 CB pointer가 다음 실행에서도 정상인지 확인합니다.
    repeat_dispatches=0
    for shape in ('128,128,128','96,64,160','65,257,97'):
        for depth in (1,2):
            name='repeat_'+shape.replace(',','_')+f'_depth{depth}'
            report=execute(name,'mcast',depth,4,'signed_random:'+shape)
            row=report['cases'][0]
            if len(report['cases'])!=1 or len(row['timing']['samples_ms'])!=5 or not row['timing']['all_repetition_outputs_equal']:
                raise RuntimeError('Repeated multicast run failed')
            repeat_dispatches+=7
    assessment={'passed':True,'total_fixture_cases':200,'repeat_profiles':6,'repeat_dispatches':repeat_dispatches,
                'coverage':coverage,'comparisons':comparisons,
                'scope':'A multicast only; B read locally; no hardware bandwidth/speedup claim'}
    (out/'design-assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    metadata['passed']=True
finally:
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
print('A multicast MatMul PASS: 200 fixture cases + 42 repeated dispatches',flush=True)
