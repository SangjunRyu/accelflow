"""Check genuine 2D A/B multicast, corner tails, degenerate grids and semaphore reuse."""
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
out=root/'results/matmul_mcast2d'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
sources=[p for p in (root/'npu/matmul').rglob('*') if p.is_file()]
sources += [Path(__file__),root/'scripts/run_mcast2d_matmul.sh',root/'npu/vector_add/common.hpp']
metadata={'date_utc':datetime.now(timezone.utc).isoformat(),'backend':'ttsim',
          'tt_metal_revision':revision,'ttsim_sha256':digest,'runs':[],'repeat_runs':[],'passed':False,
          'compiler':subprocess.check_output(['clang++-20','--version'],text=True),
          'kernel':subprocess.check_output(['uname','-r'],text=True),
          'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}

def execute(name,variant,depth,cores,selector=None):
    path=out/f'{name}.json'
    command=[str(root/'build/matmul/matmul'),str(path),variant,str(depth)]
    if variant!='baseline':
        command.append(str(cores))
    if selector:
        if variant=='baseline':
            command.append('2')
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
    if code or not path.is_file():
        raise RuntimeError(f'Failed {name}: exit {code}; see {out/name}.log')
    report=json.loads(path.read_text())
    if not report['passed']:
        raise RuntimeError('Failed CPU validation')
    print(f'PASS {name}: {len(report["cases"])} fixtures',flush=True)
    return report

def validate(row,requested,depth):
    mt,kt,nt=row['tile_grid_mkn']
    dist=row['distribution']
    rows,cols=dist['core_grid_rows_cols']
    if requested==1:
        expected_grid=(1,1)
    elif requested==2:
        expected_grid=(1,2) if nt>=2 else (min(2,mt),1)
    else:
        expected_grid=(1,min(4,nt)) if mt==1 else ((min(4,mt),1) if nt==1 else (2,2))
    if (rows,cols)!=expected_grid:
        raise RuntimeError('Unexpected core grid policy')
    p=rows*cols
    mg,ng=(mt+rows-1)//rows,(nt+cols-1)//cols
    ar,br=mt*kt*ng,nt*kt*mg
    expected={'active_cores':p,'a_dram_reads_model':ar,'b_dram_reads_model':br,
              'a_multicast_calls_model':ar if cols>1 else 0,
              'b_multicast_calls_model':br if rows>1 else 0,
              'a_receiver_deliveries_model':ar*(cols-1),'b_receiver_deliveries_model':br*(rows-1),
              'unused_tail_a_tiles_model':mt*kt*(ng*cols-nt),
              'unused_tail_b_tiles_model':nt*kt*(mg*rows-mt),
              'dummy_a_slots_model':(mg*rows-mt)*ng*cols*kt,
              'dummy_b_slots_model':(ng*cols-nt)*mg*rows*kt,
              'cb_phases_per_core_model':mg*ng*kt}
    for key,value in expected.items():
        if dist[key]!=value:
            raise RuntimeError(f'Incorrect 2D source model: {key}')
    seen=[]
    coords=set()
    origin=dist['assignments'][0]['physical_core']
    for item in dist['assignments']:
        c,r=item['logical_core']
        if (c,r) in coords or not (0<=c<cols and 0<=r<rows):
            raise RuntimeError('Duplicate/invalid core')
        coords.add((c,r))
        if item['physical_core']!=[origin[0]+c,origin[1]+r]:
            raise RuntimeError('Multicast row/column includes nonparticipant cores')
        ids=[m*nt+n for m in range(r,mt,rows) for n in range(c,nt,cols)]
        if item['output_tile_ids']!=ids or item['A_sender']!=(c==0) or item['B_sender']!=(r==0):
            raise RuntimeError('Output or sender mapping differs')
        seen+=ids
    if len(coords)!=p or sorted(seen)!=list(range(mt*nt)):
        raise RuntimeError('Output coverage/partition failed')
    expected_row={'core_count':p,'a_tile_reads_model':ar,'b_tile_reads_model':br,
                  'dram_read_payload_model':2048*(ar+br),'dram_write_payload_model':2048*mt*nt,
                  'matmul_tile_calls_model':mt*kt*nt,'read_barriers_model':ar+br,
                  'cb_capacity_bytes_model':3*depth*2048,'cb_capacity_bytes_total_model':p*3*depth*2048}
    for key,value in expected_row.items():
        if row[key]!=value:
            raise RuntimeError(f'Incorrect report: {key}')
    return rows,cols

try:
    configs=[('mcast2d',d,c) for c in (4,2,1) for d in (1,2)]
    configs += [(v,d,1 if v=='baseline' else 4) for v in ('baseline','multicore','mcast') for d in (1,2)]
    reports={}
    for variant,depth,cores in configs:
        name=f'{variant}_cores{cores}_depth{depth}'
        reports[name]=execute(name,variant,depth,cores)
        if len(reports[name]['cases'])!=20:
            raise RuntimeError('Expected 20 common fixtures')
    baseline=reports['baseline_cores1_depth2']['cases']
    comparisons=[]
    coverage={'genuine_2x2':False,'horizontal':False,'vertical':False,'single_core':False,
              'tail_M':False,'tail_N':False,'double_tail':False,'CB_wrap':False,'large_K_2x2':False}
    for index,base in enumerate(baseline):
        models={}
        for name,report in reports.items():
            row=report['cases'][index]
            for key in ('name','shape_mkn','input_fingerprint_a','input_fingerprint_b','output_fingerprint'):
                if row[key]!=base[key]:
                    raise RuntimeError(f'Controlled comparison mismatch: {name}/{key}')
            if not all(row[k] for k in ('passed','layout_matches_official_tilizer','output_sentinel_overwritten','padding_verified','fp32_destination_accumulation')):
                raise RuntimeError('Functional contract failed')
            if report['variant']!='mcast2d':
                continue
            depth=report['stream_cb_depth']
            rows,cols=validate(row,report['requested_cores'],depth)
            mt,kt,nt=row['tile_grid_mkn']
            dist=row['distribution']
            coverage['genuine_2x2'] |= rows==cols==2
            coverage['horizontal'] |= rows==1 and cols>1
            coverage['vertical'] |= rows>1 and cols==1
            coverage['single_core'] |= rows==cols==1
            coverage['tail_M'] |= mt%rows!=0
            coverage['tail_N'] |= nt%cols!=0
            coverage['double_tail'] |= mt%rows!=0 and nt%cols!=0
            coverage['CB_wrap'] |= dist['cb_phases_per_core_model']>depth
            coverage['large_K_2x2'] |= rows==cols==2 and kt==16
            models[name]={'grid_rows_cols':[rows,cols],
                          'A_reads':row['a_tile_reads_model'],'B_reads':row['b_tile_reads_model'],
                          'read_bytes_model':row['dram_read_payload_model'],
                          'reduction_vs_baseline_model':1-row['dram_read_payload_model']/base['dram_read_payload_model']}
        comparisons.append({'name':base['name'],'shape_mkn':base['shape_mkn'],
                            'controlled_outputs_match':True,'models':models})
    if not all(coverage.values()):
        raise RuntimeError(f'Incomplete coverage: {coverage}')

    repeat_dispatches=0
    for shape in ('128,128,128','96,64,160','64,512,64'):
        for depth in (1,2):
            name='repeat_'+shape.replace(',','_')+f'_depth{depth}'
            report=execute(name,'mcast2d',depth,4,'signed_random:'+shape)
            row=report['cases'][0]
            validate(row,4,depth)
            if len(report['cases'])!=1 or len(row['timing']['samples_ms'])!=5 or not row['timing']['all_repetition_outputs_equal']:
                raise RuntimeError('Repeated 2D handshake failed')
            repeat_dispatches+=7

    # 기본 20개 fixture와 별도로 1×4/4×1 배치의 물리 multicast 범위를 검증합니다.
    fallback=[]
    for shape in ('32,64,128','128,64,32'):
        for depth in (1,2):
            ref=execute('fallback_baseline_'+shape.replace(',','_')+f'_depth{depth}',
                        'baseline',depth,1,'signed_random:'+shape)['cases'][0]
            row=execute('fallback_mcast2d_'+shape.replace(',','_')+f'_depth{depth}',
                        'mcast2d',depth,4,'signed_random:'+shape)['cases'][0]
            rows,cols=validate(row,4,depth)
            for key in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint'):
                if row[key]!=ref[key]:
                    raise RuntimeError('Fallback numerical comparison failed')
            fallback.append({'shape_mkn':row['shape_mkn'],'grid_rows_cols':[rows,cols],'depth':depth,'passed':True})
            repeat_dispatches+=14
    assessment={'passed':True,'total_fixture_cases':240,'coverage':coverage,'comparisons':comparisons,
                'repeated_dispatches':repeat_dispatches,'repeat_profiles':14,'fallback_comparisons':fallback,
                'scope':'real A row multicast and B column multicast; no hardware speedup claim'}
    assert repeat_dispatches==98
    (out/'design-assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    metadata['passed']=True
finally:
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
print('2D multicast MatMul PASS: 240 fixture cases + 98 repeated/fallback dispatches',flush=True)
