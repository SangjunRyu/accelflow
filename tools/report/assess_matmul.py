"""Validate controlled fixtures and independently check K-streaming design models."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
paths = sorted((root / 'results/matmul').glob('*/metadata.json'))
if not paths:
    raise SystemExit('No MatMul suite')
directory = paths[-1].parent
metadata = json.loads(paths[-1].read_text())
if not metadata.get('passed'):
    raise SystemExit('Latest suite failed; refusing to assess old success')

expected_modes = {f'{v}_depth{d}' for v in ('baseline', 'reuse_a', 'blocked') for d in (1, 2)}
expected_modes |= {f'streamed_depth{d}_chunk{c}' for d in (1, 2) for c in (1, 2)}
reports, rows_by_mode = {}, {}
def fixture_key(row):
    return row['name'], tuple(row['shape_mkn'])

for run in metadata['runs']:
    name = Path(run['report']).stem
    report = json.loads((directory / run['report']).read_text())
    if run['exit_code'] != 0 or not report.get('passed'):
        raise SystemExit(f'Failed configuration: {name}')
    reports[name] = report
    rows = {fixture_key(row): row for row in report['cases']}
    if len(rows) != len(report['cases']):
        raise SystemExit('Duplicate fixtures')
    rows_by_mode[name] = rows
if set(reports) != expected_modes:
    raise SystemExit('Expected all ten configurations')

reference = reports['baseline_depth2']['cases']
if len(reference) != 20:
    raise SystemExit('Expected 14 common and 6 extended fixtures')
comparisons = []
for baseline in reference:
    key = fixture_key(baseline)
    mt, kt, nt = baseline['tile_grid_mkn']
    products = mt * kt * nt
    mb, nb = (mt + 1) // 2, (nt + 1) // 2
    cb_models = {}

    for mode, report in reports.items():
        row = rows_by_mode[mode].get(key)
        variant, depth = report['variant'], report['stream_cb_depth']
        chunk = report['k_chunk_tiles']
        if row is None:
            if variant == 'blocked' and kt > 4:
                continue
            raise SystemExit(f'Missing fixture: {mode}/{key}')
        if not all(row[k] for k in ('passed', 'padding_verified', 'output_sentinel_overwritten', 'layout_matches_official_tilizer')):
            raise SystemExit(f'Functional contract failed: {mode}')
        for field in ('input_fingerprint_a', 'input_fingerprint_b', 'output_fingerprint'):
            if row[field] != baseline[field]:
                raise SystemExit(f'Controlled comparison differs: {mode}/{field}/{key}')

        chunks = (kt + chunk - 1) // chunk if variant == 'streamed' else 1
        if variant in ('blocked', 'streamed'):
            block_k = chunk if variant == 'streamed' else kt
            a_reads, b_reads = mt * kt * nb, nt * kt * mb
            capacity = [2 * block_k * depth, 2 * block_k * depth, depth]
            barriers, registers = mb * nb * chunks, 4
        elif variant == 'reuse_a':
            a_reads, b_reads = mt * kt, products
            capacity = [kt, depth, depth]
            barriers, registers = mt + products, 1
        else:
            a_reads = b_reads = products
            capacity = [depth] * 3
            barriers, registers = 2 * products, 1
        expected = {
            'a_tile_reads_model': a_reads, 'b_tile_reads_model': b_reads,
            'dram_read_payload_model': 2048 * (a_reads + b_reads),
            'output_tile_writes_model': mt * nt,
            'dram_write_payload_model': 2048 * mt * nt,
            'matmul_tile_calls_model': products,
            'cb_capacity_tiles': capacity, 'cb_capacity_bytes_model': 2048 * sum(capacity),
            'read_barriers_model': barriers, 'destination_tile_slots_max': registers,
            'input_chunks_per_output_block': chunks, 'partial_result_pack_calls_model': 0,
            'fp32_destination_accumulation': True,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise SystemExit(f'Model mismatch: {mode}/{field}')
        cb_models[mode] = row['cb_capacity_bytes_model']

    streamed = rows_by_mode['streamed_depth2_chunk2'][key]
    block = rows_by_mode['blocked_depth2'].get(key)
    comparisons.append({
        'name': baseline['name'], 'shape_mkn': baseline['shape_mkn'],
        'tile_grid_mkn': [mt, kt, nt], 'same_input_output_fingerprints': True,
        'configs_compared': len(cb_models), 'old_blocked_skipped': block is None,
        'baseline_read_bytes_model': baseline['dram_read_payload_model'],
        'streamed_read_bytes_model': streamed['dram_read_payload_model'],
        'read_reduction_vs_baseline_model': 1 - streamed['dram_read_payload_model'] / baseline['dram_read_payload_model'],
        'cb_bytes_by_configuration': cb_models,
        'streamed_chunks_per_output_block_chunk2': (kt + 1) // 2,
        'partial_K_chunk2': bool(kt % 2),
        'partial_output_block': bool(mt % 2 or nt % 2),
        'max_abs_error': streamed['max_abs_error'],
        'metric_kind': 'source-derived model; no hardware counters or latency',
    })

reference_keys = {fixture_key(row) for row in reference}
for mode, report in reports.items():
    allowed = {fixture_key(row) for row in reference if report['variant'] != 'blocked' or row['tile_grid_mkn'][1] <= 4}
    if set(rows_by_mode[mode]) != allowed:
        raise SystemExit(f'Unexpected fixtures in {mode}')
    if report['variant'] == 'streamed':
        sizes = {row['cb_capacity_bytes_model'] for row in report['cases']}
        if len(sizes) != 1:
            raise SystemExit('Streamed CB capacity unexpectedly depends on K')

assessment = {
    'passed': True, 'total_cases': sum(len(r['cases']) for r in reports.values()),
    'evidence_directory': str(directory.relative_to(root)), 'comparisons': comparisons,
    'verdict': {
        'cpu_reference_layout_padding_sentinel': 'all cases pass',
        'controlled_comparison': 'same input/output FNV fingerprints for every supported configuration',
        'K_streaming': 'K=129/257/512 tested; chunk1/2 and depth1/2 pass',
        'partial_K_and_input_CB_wrap': 'functional fixtures pass',
        'constant_CB_capacity': 'configuration model independent of total K',
        'partial_result_spill': 'none in source; destination tiles retained across chunks',
        'hardware_latency_bandwidth_overlap': 'not measured',
    },
}
assert assessment['total_cases'] == 188
(directory / 'design-assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
lines = ['# K-streamed MatMul assessment', '', f'Evidence: `{directory.relative_to(root)}`', '',
         '| M,K,N | Configs | K chunks at chunk2 | Last partial K chunk | Baseline/streamed read bytes | CB depth2 blocked/streamed chunk2 |',
         '| --- | --- | --- | --- | --- | --- |']
for row in comparisons:
    sizes = row['cb_bytes_by_configuration']
    lines.append(f'| {row["shape_mkn"]} | {row["configs_compared"]} | '
                 f'{row["streamed_chunks_per_output_block_chunk2"]} | {row["partial_K_chunk2"]} | '
                 f'{row["baseline_read_bytes_model"]}/{row["streamed_read_bytes_model"]} | '
                 f'{sizes.get("blocked_depth2", "not supported")}/{sizes["streamed_depth2_chunk2"]} |')
lines += ['', 'Counts are source-derived models. Hardware speedup and overlap are not measured.', '']
(directory / 'design-assessment.md').write_text('\n'.join(lines))
print(f'MatMul K-streaming assessment PASS: {assessment["total_cases"]} cases; evidence={directory}')
