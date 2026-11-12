"""Independently audit benchmark raw medians, paired data and CPU FP64 outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import struct


def load(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    out=parser.parse_args().results.resolve()
    meta=load(out/'metadata.json');summary=load(out/'summary.json')
    assert meta['passed'] and summary['passed'] and len(meta['runs'])==24
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((out/'sources'/name).read_bytes()).hexdigest()==digest
    fixtures={f['key']:f for f in load(out/'selected-fixtures.json')['fixtures']}
    references={}
    for key,f in fixtures.items():
        raw=(out/f['binary']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256']
        magic,h,w,hp,wp,eps=struct.unpack_from('<8sIIIIf',raw)
        assert magic==b'AFRM0001' and len(raw)==28+(hp*wp+wp)*2
        words=struct.unpack_from('<'+'H'*(hp*wp+wp),raw,28)
        x=[struct.unpack('<f',struct.pack('<I',b<<16))[0] for b in words]
        reference=[]
        for row in range(h):
            stat=math.fsum(float(x[row*wp+col])**2 for col in range(w))
            inv=1/math.sqrt(stat/w+eps)
            reference.extend(x[row*wp+col]*inv*x[hp*wp+col] for col in range(w))
        references[key]=reference
    rows=[];groups={};hashes={}
    for run in meta['runs']:
        assert run['exit_code']==0
        report=out/(run['label']+'.json');hashes[report.name]=hashlib.sha256(report.read_bytes()).hexdigest()
        data=load(report);assert data['passed'] and len(data['cases'])==1
        case=data['cases'][0];t=case['benchmark_timing'];key=run['key'];variant=run['variant']
        assert t['sample_count']==5 and t['warmup_count']==2 and t['physical_npu_latency'] is False
        assert case['repeats']==7 and len(t['samples_ms'])==5 and len(t['stage_samples_ms'])==5
        assert len(case['logical_output'])==len(references[key])
        error=max(abs(got-ref) for got,ref in zip(case['logical_output'],references[key]))
        assert all(math.isfinite(got) and abs(got-ref)<=1e-4+1e-4*abs(ref)
                   for got,ref in zip(case['logical_output'],references[key]))
        groups.setdefault((key,variant),[]).append(median(t['samples_ms']))
        rows.append(dict(key=key,variant=variant,trial=run['trial'],passed=True,max_abs_error_vs_python_fp64=error))
    for row in summary['results']:
        for variant in ('split','compat'):
            assert median(groups[row['key'],variant])==row[variant]['median_ms']
        ratio=row['split']['median_ms']/row['compat']['median_ms']
        assert math.isclose(ratio,row['split_over_compat_ratio'],rel_tol=1e-12)
    result=dict(passed=True,process_reports=24,independent_python_reference_outputs=24,
        max_abs_error_vs_python_fp64=max(r['max_abs_error_vs_python_fp64'] for r in rows),
        source_snapshots_verified=True,raw_medians_verified=True,rows=rows,report_sha256=hashes,
        scope='Saved-output CPU reference audit; TTSim timing is not physical-chip performance')
    (out/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('RMSNORM_TIMING_AUDIT_ALL_PASS',flush=True)


if __name__=='__main__':main()
