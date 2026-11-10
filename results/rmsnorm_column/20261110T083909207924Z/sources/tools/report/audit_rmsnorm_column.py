"""Audit saved row/column-sharded RMSNorm against PyTorch CPU FP32 and FP64."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import torch
import torch.nn.functional as functional


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    out=parser.parse_args().results.resolve()
    assert load(out/'summary.json')['passed']
    labels=['fused1','fused2','fused4','column1','column2','column4','column4-depth1']
    reports={label:{c['key']:c for c in load(out/(label+'.json'))['cases']} for label in labels}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[]
    with torch.inference_mode():
        for fixture in load(out/'fixtures.json')['fixtures']:
            raw=(out/fixture['binary']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==fixture['sha256']
            magic,h,w,hp,wp,epsilon=struct.unpack_from('<8sIIIIf',raw)
            assert magic==b'AFRM0001' and len(raw)==28+(hp*wp+wp)*2
            data=torch.frombuffer(bytearray(raw[28:]),dtype=torch.bfloat16).float()
            x=data[:hp*wp].reshape(hp,wp)[:h,:w].contiguous();gamma=data[hp*wp:][:w].contiguous()
            refs={'fp32':functional.rms_norm(x,(w,),weight=gamma,eps=epsilon).double(),
                  'fp64':functional.rms_norm(x.double(),(w,),weight=gamma.double(),eps=epsilon)}
            for label in labels:
                case=reports[label][fixture['key']]
                assert case['passed'] and case['padding_zero'] and case['assignment_has_no_gaps_or_overlaps']
                actual=torch.tensor(case['logical_output'],dtype=torch.float64).reshape(h,w)
                assert bool(torch.isfinite(actual).all());errors={}
                for dtype,reference in refs.items():
                    difference=(actual-reference).abs()
                    assert bool((difference<=1e-4+1e-4*reference.abs()).all()),(fixture['key'],label,dtype)
                    errors[dtype]=difference.max().item()
                rows.append(dict(key=fixture['key'],label=label,passed=True,max_abs_errors=errors))
            print('COLUMN_PYTORCH_PASS '+fixture['key'],flush=True)
    meta=load(out/'metadata.json');assert meta['passed']
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((out/'sources'/name).read_bytes()).hexdigest()==digest
    result=dict(passed=True,cases=len(load(out/'fixtures.json')['fixtures']),backend_outputs_compared=len(rows),
        torch_version=torch.__version__,command=[sys.executable,*sys.argv],atol=1e-4,rtol=1e-4,
        max_abs_error_vs_pytorch_fp32=max(r['max_abs_errors']['fp32'] for r in rows),
        max_abs_error_vs_pytorch_fp64=max(r['max_abs_errors']['fp64'] for r in rows),rows=rows,
        report_sha256={label+'.json':hashlib.sha256((out/(label+'.json')).read_bytes()).hexdigest() for label in labels},
        scope='Finite BF16 corpus with explicit serialized epsilon; correctness only, not real chip performance')
    (out/'pytorch-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COLUMN_PYTORCH_ALL_PASS outputs='+str(len(rows)),flush=True)


if __name__=='__main__':main()
