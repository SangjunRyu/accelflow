"""Recalculate the public portfolio tables from committed experiment records.

This audits saved evidence. It does not execute CUDA or simulate a device.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(root):
    sources = {}

    def read(relative):
        raw = (root / relative).read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def same(actual, expected):
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6),
                f'Raw samples disagree with summary: {actual} vs {expected}')

    design_inputs = [
        ('baseline', 'results/matmul/20261109T075137507595Z/baseline_depth2.json'),
        ('reuse_a', 'results/matmul/20261109T075137507595Z/reuse_a_depth2.json'),
        ('blocked', 'results/matmul/20261109T075137507595Z/blocked_depth2.json'),
        ('multicore', 'results/matmul_mcast2d/20261019T111449423414Z/multicore_cores4_depth2.json'),
        ('mcast', 'results/matmul_mcast2d/20261019T111449423414Z/mcast_cores4_depth2.json'),
        ('mcast2d', 'results/matmul_mcast2d/20261019T111449423414Z/mcast2d_cores4_depth2.json'),
        ('mcast2d_batch4', 'results/matmul_batched_perf/20261019T113209749850Z/correctness_b4_d2_p4.json'),
    ]
    design = []
    signatures = []
    for variant, path in design_inputs:
        data = read(path)
        require(data['passed'], path)
        matches = [c for c in data['cases'] if c['shape_mkn'] == [128, 128, 128]]
        require(len(matches) == 1, path)
        case = matches[0]
        require(case['passed'] and case['fp32_destination_accumulation'] and
                case['layout_matches_official_tilizer'] and case['padding_verified'] and
                case['output_sentinel_overwritten'] and case['max_abs_error'] == 0, path)
        signatures.append([case[k] for k in ('input_fingerprint_a', 'input_fingerprint_b', 'output_fingerprint')])
        design.append(dict(variant=variant, evidence_kind='source_model', shape_mkn=[128,128,128],
            active_cores=case['core_count'], dram_read_bytes=case['dram_read_payload_model'],
            cb_bytes_per_core=case['cb_capacity_bytes_model'],
            total_cb_bytes=case['cb_capacity_bytes_total_model'], source=path))
    require(all(s == signatures[0] for s in signatures), 'Design cases use different inputs or outputs')

    batched = 'results/matmul_batched_perf/20261019T113209749850Z'
    batch_meta = read(batched + '/metadata.json')
    require(batch_meta['passed'] and batch_meta['trials'] == 3 and batch_meta['samples'] == 5 and
            batch_meta['warmups'] == 2, 'Unexpected TTSim measurement conditions')
    ttsim = []
    per_shape = {}
    for row in read(batched + '/summary.json'):
        shape, variant = row['shape'], row['variant']
        trials = []
        case_signatures = []
        case = None
        for trial in (1, 2, 3):
            path = batched + f'/trial{trial}_' + 'x'.join(map(str,shape)) + f'_{variant}.json'
            data = read(path)
            require(data['passed'] and len(data['cases']) == 1, path)
            case = data['cases'][0]
            timing = case['timing']
            require(case['passed'] and case['shape_mkn'] == shape and case['core_count'] == 4 and
                    timing['all_repetition_outputs_equal'] and not timing['physical_NPU_latency'] and
                    len(timing['samples_ms']) == 5 and len(timing['warmup_ms']) == 2, path)
            trials.append(median(timing['samples_ms']))
            case_signatures.append([case[k] for k in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint')])
        require(all(s == case_signatures[0] for s in case_signatures), 'TTSim fingerprints changed across trials')
        previous = per_shape.setdefault(tuple(shape), case_signatures[0])
        require(previous == case_signatures[0], 'TTSim fingerprints differ across variants')
        same(median(trials), row['median_ms'])
        for a,b in zip(trials,row['trial_medians_ms']):same(a,b)
        require(row['dram_read_bytes'] == case['dram_read_payload_model'], 'DRAM model mismatch')
        distribution = case['distribution']
        ttsim.append(dict(evidence_kind='ttsim_host_observation', shape_mkn=shape, variant=variant,
            median_ms=median(trials), trial_medians_ms=trials,
            dram_read_bytes_model=case['dram_read_payload_model'],
            cb_bytes_per_core_model=case['cb_capacity_bytes_model'],
            ready_atomics_model=distribution.get('ready_atomic_increments_model'),
            payload_multicast_calls_model=(distribution['a_multicast_calls_model'] + distribution['b_multicast_calls_model'])
                if 'b_multicast_calls_model' in distribution else distribution.get('a_multicast_calls_model',0),
            source=batched+'/summary.json'))

    cuda_dir = 'results/cuda_matmul/20261109T0747432221509Z'
    cuda_meta = read(cuda_dir + '/cuda-metadata.json')
    require(cuda_meta['passed'], 'CUDA metadata failed')
    cuda = []
    for row in read(cuda_dir + '/cuda-summary.json'):
        fixture, variant = row['fixture'], row['variant']
        trials = []
        npu = read(cuda_dir + f'/npu_{fixture}.json')
        require(npu['passed'] and len(npu['cases']) == 1, 'NPU input export failed')
        npu_case = npu['cases'][0]
        for trial in (1,2,3):
            path = cuda_dir + f'/cuda_{fixture}_{variant}_trial{trial}.json'
            case = read(path)
            require(case['passed'] and case['gpu'] == 'NVIDIA GeForce MX150' and
                    case['shape_mkn'] == row['shape'] and case['samples'] == 15 and
                    case['warmups'] == 20 and case['launches_per_event_sample'] == 20 and
                    len(case['kernel_event_batch_average_ms']) == 15, path)
            for key in ('input_fingerprint_a','input_fingerprint_b'):
                require(case[key] == npu_case[key], 'CUDA and NPU input fingerprints differ')
            require(case['bf16_output_fingerprint'] == npu_case['output_fingerprint'],
                    'Representative integer CUDA/NPU outputs differ')
            trials.append(median(case['kernel_event_batch_average_ms']))
        same(median(trials),row['medians_ms']['kernel_event_batch_average_ms'])
        for a,b in zip(sorted(trials),sorted(row['trial_kernel_medians_ms'])):same(a,b)
        cuda.append(dict(evidence_kind='cuda_gpu_measurement',shape_mkn=row['shape'],variant=variant,
            event_batch_average_median_ms=median(trials),trial_medians_ms=trials,
            source=cuda_dir+'/cuda-summary.json'))
    return dict(schema_version=1,passed=True,scope='Audit of saved records; no kernel execution',
        design_models=design,design_fingerprints=signatures[0],
        cuda_measurements=cuda,ttsim_observations=ttsim,
        source_sha256=dict(sorted(sources.items())))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    data=collect(root)
    args.write.parent.mkdir(parents=True,exist_ok=True)
    args.write.write_text(json.dumps(data,indent=2)+'\n')
    print(f'PORTFOLIO_EVIDENCE_PASS sources={len(data["source_sha256"])}')
