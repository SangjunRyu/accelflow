"""Verify independent-core tile partitioning against the existing single-core baseline."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
metal = Path(os.environ.get('TT_METAL_HOME', str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
sim = Path(os.environ.get('TT_METAL_SIMULATOR', str(Path.home()/'dev/tenstorrent/sim/libttsim_wh.so'))).resolve()
revision = subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
digest = hashlib.sha256(sim.read_bytes()).hexdigest()
if revision != 'de546d3b146758714d900f11b218c8f9c805f410':
    raise SystemExit('Expected pinned TT-Metal v0.79.0')
if digest != '196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a':
    raise SystemExit('Expected pinned TTSim v1.11.1')
(sim.parent/'soc_descriptor.yaml').write_bytes((metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml').read_bytes())
env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
           TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE='1')
out = root/'results/matmul_multicore'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
sources = [p for p in (root/'npu/matmul').rglob('*') if p.is_file()]
sources += [Path(__file__),root/'tools/report/build_matmul.py',root/'scripts/run_multicore_matmul.sh',root/'npu/vector_add/common.hpp']
metadata = {
    'date_utc': datetime.now(timezone.utc).isoformat(), 'backend':'ttsim',
    'tt_metal_revision': revision, 'ttsim_sha256':digest,
    'compiler':subprocess.check_output(['clang++-20','--version'],text=True).strip(),
    'kernel':subprocess.check_output(['uname','-r'],text=True).strip(),
    'os_release':Path('/etc/os-release').read_text(),
    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'runs':[], 'passed':False,
    'policy':'Functional output partitioning and source-derived models; no hardware speedup claim',
}
reports = {}
try:
    configs = [('baseline',d,1) for d in (1,2)]
    configs += [('multicore',d,c) for c in (1,2,4) for d in (1,2)]
    for variant,depth,cores in configs:
        name = f'{variant}_cores{cores}_depth{depth}'
        path = out/f'{name}.json'
        command = [str(root/'build/matmul/matmul'),str(path),variant,str(depth)]
        if variant == 'multicore':
            command.append(str(cores))
        with (out/f'{name}.log').open('w') as log:
            log.write('Command: '+repr(command)+'\n'); log.flush()
            try:
                code = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
            except subprocess.TimeoutExpired:
                code = 124
            log.write(f'example_exit_code={code}\n')
        metadata['runs'].append({'name':name,'variant':variant,'cores':cores,'depth':depth,'exit_code':code})
        if code != 0 or not path.is_file():
            raise RuntimeError(f'Failed {name}: {out/name}.log')
        report = json.loads(path.read_text())
        if not report['passed'] or len(report['cases']) != 20:
            raise RuntimeError('Expected all 20 validated cases')
        reports[name] = report
        print(f'PASS {name}: 20 fixtures',flush=True)

    reference = reports['baseline_cores1_depth2']['cases']
    comparisons = []
    coverage = {'more_requested_than_tiles':False, 'uneven_partition':False,
                'range_crosses_output_row':False, 'CB_wrap_per_core':False,
                'four_active_cores':False, 'large_K_512':False}
    for index, baseline in enumerate(reference):
        mt,kt,nt = baseline['tile_grid_mkn']
        total = mt*nt
        partitions = {}
        for name,report in reports.items():
            row = report['cases'][index]
            for key in ('name','shape_mkn','input_fingerprint_a','input_fingerprint_b','output_fingerprint'):
                if row[key] != baseline[key]:
                    raise RuntimeError(f'Controlled comparison mismatch: {name}/{key}')
            if not all(row[key] for key in ('passed','padding_verified','layout_matches_official_tilizer','output_sentinel_overwritten','fp32_destination_accumulation')):
                raise RuntimeError('Functional contract failed')
            for key in ('a_tile_reads_model','b_tile_reads_model','dram_read_payload_model','dram_write_payload_model','matmul_tile_calls_model','read_barriers_model'):
                if row[key] != baseline[key]:
                    raise RuntimeError(f'Baseline traffic/control model changed: {key}')
            depth = report['stream_cb_depth']
            if row['cb_capacity_bytes_model'] != 3*depth*2048:
                raise RuntimeError('Per-core CB capacity mismatch')
            distribution = row['distribution']
            if distribution is None:
                continue
            requested = report['requested_cores']
            active = min(requested,total)
            assignments = distribution['assignments']
            if len(assignments) != active or distribution['active_cores'] != active or row['core_count'] != active:
                raise RuntimeError('Active core count mismatch')
            grid_x,grid_y = distribution['grid']
            cursor = 0
            coordinates = set()
            counts = []
            for core_index,item in enumerate(assignments):
                start,count = item['start_tile'],item['tile_count']
                x,y = item['logical_core']
                if start != cursor or count != total//active+(core_index<total%active):
                    raise RuntimeError('Gap, overlap or unbalanced partition')
                if not (0<=x<grid_x and 0<=y<grid_y) or (x,y) in coordinates:
                    raise RuntimeError('Invalid or duplicate logical core')
                coordinates.add((x,y))
                counts.append(count)
                cursor += count
                coverage['range_crosses_output_row'] |= start//nt != (cursor-1)//nt
                coverage['CB_wrap_per_core'] |= count > depth
            if cursor != total or max(counts)-min(counts)>1:
                raise RuntimeError('Output coverage or balance failed')
            if row['cb_capacity_bytes_total_model'] != active*3*depth*2048:
                raise RuntimeError('Aggregate CB capacity mismatch')
            coverage['more_requested_than_tiles'] |= requested>total
            coverage['uneven_partition'] |= total%active != 0
            coverage['four_active_cores'] |= active==4
            coverage['large_K_512'] |= row['shape_mkn'][1]==512
            partitions[name] = distribution
        comparisons.append({'name':baseline['name'],'shape_mkn':baseline['shape_mkn'],
                            'output_tiles':total,'input_output_fingerprints_match':True,
                            'global_DRAM_payload_matches_baseline_model':True,'partitions':partitions})
    if not all(coverage.values()):
        raise RuntimeError(f'Missing coverage: {coverage}')
    assessment = {'passed':True,'total_cases':sum(len(r['cases']) for r in reports.values()),
                  'coverage':coverage,'comparisons':comparisons,
                  'performance':'Not measured; independent-core mapping does not reduce global input payload'}
    assert assessment['total_cases'] == 160
    (out/'design-assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    metadata['passed'] = True
finally:
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Evidence:',out,flush=True)
print('Independent multicore MatMul PASS: 160 cases / 8 configurations',flush=True)
