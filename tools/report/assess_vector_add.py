"""Assess design contracts from real outputs, runtime buffer observations and static models."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
evidence = []
for path in sorted((root/'results/vector_add').glob('*/metadata.json')):
    metadata = json.loads(path.read_text())
    if metadata.get('passed') and {r['variant'] for r in metadata['runs']} == {'riscv','tiled_depth1','tiled'}:
        evidence.append(path.parent)
if not evidence: raise SystemExit('No successful controlled CB-depth experiment')
directory = evidence[-1]
scalar = json.loads((directory/'riscv.json').read_text())
depth1 = json.loads((directory/'tiled_depth1.json').read_text())
depth2 = json.loads((directory/'tiled.json').read_text())
if len(depth1['cases']) != len(depth2['cases']): raise SystemExit('Different test sets')
if not scalar['cases'] or not depth1['cases'] or not all(c['passed'] for c in scalar['cases']):
    raise SystemExit('Missing or failing scalar cases')
models = []
controlled = []
for a, b in zip(depth1['cases'],depth2['cases']):
    if a['cb_tiles_each'] != 1 or b['cb_tiles_each'] != 2 or \
       a['logical_cb_capacity_bytes'] != 6144 or b['logical_cb_capacity_bytes'] != 12288:
        raise SystemExit('Unexpected circular-buffer configuration')
    same_input = all(a[k] == b[k] for k in ('name','length','input_fingerprint_a','input_fingerprint_b'))
    if not same_input or not a['passed'] or not b['passed'] or not a['padding_verified'] or not b['padding_verified']:
        raise SystemExit('Controlled experiment or correctness contract failed')
    same_traffic = all(a[k] == b[k] for k in ('dram_read_payload_bytes','dram_write_payload_bytes','tile_count'))
    if not same_traffic: raise SystemExit('CB-only change unexpectedly changes payload model')
    n,p,t = b['length'],b['padded_length'],b['tile_count']
    models.append({
        'name':b['name'],'length':n,'padded_length':p,'tiles':t,'dtype':'bf16',
        'tile_occupancy':n/p,'padding_elements':p-n,'payload_bytes':6*p,
        'useful_payload_bytes':6*n,'padding_payload_amplification':p/n,
        'transfer_api_calls':3*t,'read_barrier_calls':t,'write_barrier_calls':t,
        'reader_cb_calls':4*t,'compute_cb_calls':6*t,'writer_cb_calls':2*t,'tile_register_control_calls':4*t,
        'useful_adds_per_payload_byte':n/(6*p),
        'metric_kind':'code-derived model, not measured packets, stalls or cycles'})
    controlled.append({'name':b['name'],'length':n,'same_input_fingerprints':same_input,
        'depth1_passed':True,'depth2_passed':True,'same_payload_model':same_traffic,
        'depth1_cb_bytes':a['logical_cb_capacity_bytes'],'depth2_cb_bytes':b['logical_cb_capacity_bytes'],
        'depth1_capacity_reuse_required':t>1,'depth2_capacity_reuse_required':t>2,
        'peak_overlap_or_stall_reduction_measured':False})
scalar_models = [{
    'name':c['name'],'length':c['length'],'dtype':'uint32','payload_bytes':12*c['length'],
    'transfer_api_calls':3*c['length'],'read_barrier_calls':c['length'],
    'store_visibility_checks':c['length'],'write_barrier_calls':c['length'],
    'useful_adds_per_payload_byte':1/12,
    'metric_kind':'code-derived model, not measured packets, stalls or cycles'} for c in scalar['cases']]
max_tiles = max(c['tile_count'] for c in depth2['cases'])
reuse_verdict = (f'pass up to {max_tiles} tiles with depth 1 and depth 2'
                 if max_tiles > 2 else 'insufficient coverage beyond depth 2')
report = {
    'evidence_directory':str(directory.relative_to(root)),
    'evidence_classes':{
        'observed':'output correctness, timeout/exit codes, buffer API values, same input fingerprints',
        'derived':'payload, high-level transfer/synchronization API counts, padding occupancy',
        'not_measured':'silicon latency, bandwidth, actual NoC packets, stalls, temporal overlap'},
    'scalar_case_count':len(scalar['cases']),'depth1_case_count':len(depth1['cases']),
    'depth2_case_count':len(depth2['cases']), 'controlled_cb_depth_comparison':controlled,
    'riscv_models':scalar_models,'tiled_models':models,
    'design_verdict':{
        'numerical_outputs':'pass in all tested cases',
        'multi_tile_capacity_reuse':reuse_verdict,
        'padding_integrity':'pass in all tested cases',
        'cb_depth2_dram_traffic_reduction':'not supported; same payload as depth1',
        'cb_depth2_temporal_overlap_or_speedup':'not established without execution timeline/hardware',
        'all_possible_interleavings':'not established by finite functional tests'},
    'passed':True
}
(directory/'design-assessment.json').write_text(json.dumps(report,indent=2)+'\n')
lines = ['# Vector Add design assessment','',f'Evidence: `{directory.relative_to(root)}`','',
         '| Implementation | Passed cases |','| --- | --- |',
         f'| uint32 RISC-V | {len(scalar["cases"])} |',
         f'| BF16 CB depth 1 | {len(depth1["cases"])} |',
         f'| BF16 CB depth 2 | {len(depth2["cases"])} |','',
         'Counts below are analytical API/payload models, not profiler counters.','',
         '| BF16 N | Tiles | Occupancy | Padding amplification | Transfer API calls | Read/write barriers |',
         '| --- | --- | --- | --- | --- | --- |']
seen=set()
for model in models:
    n=model['length']
    if n in seen: continue
    seen.add(n)
    lines.append(f'| {n} | {model["tiles"]} | {model["tile_occupancy"]:.3%} | '
                 f'{model["padding_payload_amplification"]:.4f}x | {model["transfer_api_calls"]} | '
                 f'{model["read_barrier_calls"]+model["write_barrier_calls"]} |')
lines.extend(['','CB depth 1: 6144 B; depth 2: 12288 B; payload unchanged.',
              'Correctness and CB-slot reuse passed. Actual overlap, stalls and hardware speedup remain unmeasured.',''])
(directory/'design-assessment.md').write_text('\n'.join(lines))
print('Design assessment: PASS; output:',directory/'design-assessment.json')
