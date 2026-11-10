"""Independently compare 1/2/4-core RMSNorm with official PyTorch CPU references."""
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
    args=parser.parse_args();out=args.results.resolve()
    assert load(out/'summary.json')['passed']
    manifest=load(out/'fixtures.json')
    reports={n:{c['key']:c for c in load(out/f'cores{n}.json')['cases']} for n in (1,2,4)}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[]
    with torch.inference_mode():
        for fixture in manifest['fixtures']:
            raw=(out/fixture['binary']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==fixture['sha256']
            magic,h,w,hp,wp,epsilon=struct.unpack_from('<8sIIIIf',raw)
            assert magic==b'AFRM0001' and len(raw)==28+(hp*wp+wp)*2
            data=torch.frombuffer(bytearray(raw[28:]),dtype=torch.bfloat16).float()
            x=data[:hp*wp].reshape(hp,wp)[:h,:w].contiguous()
            gamma=data[hp*wp:][:w].contiguous()
            refs={'fp32':functional.rms_norm(x,(w,),weight=gamma,eps=epsilon).double(),
                  'fp64':functional.rms_norm(x.double(),(w,),weight=gamma.double(),eps=epsilon)}
            errors={}
            single=reports[1][fixture['key']]
            single_bits=struct.pack('<'+'f'*len(single['logical_output']),*single['logical_output'])
            for cores in (1,2,4):
                case=reports[cores][fixture['key']]
                assert case['passed'] and case['assignment_has_no_gaps_or_overlaps']
                values=case['logical_output']
                assert struct.pack('<'+'f'*len(values),*values)==single_bits
                actual=torch.tensor(values,dtype=torch.float64).reshape(h,w)
                errors[str(cores)]={}
                for dtype,reference in refs.items():
                    difference=(actual-reference).abs()
                    assert bool(torch.isfinite(actual).all())
                    assert bool((difference<=1e-4+1e-4*reference.abs()).all()),(fixture['key'],cores,dtype)
                    errors[str(cores)][dtype]=difference.max().item()
            rows.append(dict(key=fixture['key'],passed=True,bitwise_equal_across_core_counts=True,max_abs_errors=errors))
            print('MULTICORE_PYTORCH_PASS '+fixture['key'],flush=True)
    assert len(rows)==5
    meta=load(out/'metadata.json')
    assert meta['passed']
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((out/'sources'/name).read_bytes()).hexdigest()==digest
    result=dict(passed=True,cases=5,backend_outputs_compared=15,torch_version=torch.__version__,device='cpu',
                command=[sys.executable,*sys.argv],atol=1e-4,rtol=1e-4,
                max_abs_error_vs_pytorch_fp32=max(r['max_abs_errors'][str(c)]['fp32'] for r in rows for c in (1,2,4)),
                max_abs_error_vs_pytorch_fp64=max(r['max_abs_errors'][str(c)]['fp64'] for r in rows for c in (1,2,4)),
                bitwise_equal_across_core_counts=True,rows=rows,
                report_sha256={f'cores{c}.json':hashlib.sha256((out/f'cores{c}.json').read_bytes()).hexdigest() for c in (1,2,4)},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Finite BF16 corpus, FP32 output; correctness only, not PyTorch or real-NPU performance benchmarking')
    (out/'pytorch-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('MULTICORE_PYTORCH_ALL_PASS cases=5 outputs=15',flush=True)


if __name__=='__main__':main()
