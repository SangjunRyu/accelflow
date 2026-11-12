"""Audit BF16 and true FP32 no-splitting RMSNorm outputs against PyTorch CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import torch
import torch.nn.functional as functional


def load(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    out=parser.parse_args().results.resolve()
    assert load(out/'summary.json')['passed']
    reports={d:{c['key']:c for c in load(out/f'compat-depth{d}.json')['cases']} for d in (1,2)}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[]
    with torch.inference_mode():
        for f in load(out/'fixtures.json')['fixtures']:
            raw=(out/f['binary']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256']
            magic,h,w,hp,wp,eps=struct.unpack_from('<8sIIIIf',raw)
            assert magic in (b'AFRM0001',b'AFRF0001')
            element_bytes=2 if magic==b'AFRM0001' else 4
            assert len(raw)==28+(hp*wp+wp)*element_bytes
            dtype=torch.bfloat16 if element_bytes==2 else torch.float32
            values=torch.frombuffer(bytearray(raw[28:]),dtype=dtype).float()
            x=values[:hp*wp].reshape(hp,wp)[:h,:w].contiguous();gamma=values[hp*wp:][:w].contiguous()
            refs={'fp32':functional.rms_norm(x,(w,),weight=gamma,eps=eps).double(),
                  'fp64':functional.rms_norm(x.double(),(w,),weight=gamma.double(),eps=eps)}
            for depth in (1,2):
                case=reports[depth][f['key']]
                actual=torch.tensor(case['logical_output'],dtype=torch.float64).reshape(h,w)
                errors={}
                assert case['passed'] and bool(torch.isfinite(actual).all())
                for name,reference in refs.items():
                    diff=(actual-reference).abs();assert bool((diff<=1e-4+1e-4*reference.abs()).all()),(f['key'],depth,name)
                    errors[name]=diff.max().item()
                rows.append(dict(key=f['key'],fixture_dtype=f['fixture_dtype'],depth=depth,passed=True,max_abs_errors=errors))
            print('FP32_DEST_PYTORCH_PASS '+f['key'],flush=True)
    meta=load(out/'metadata.json');assert meta['passed']
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((out/'sources'/name).read_bytes()).hexdigest()==digest
    result=dict(passed=True,fixtures=29,backend_outputs_compared=len(rows),torch_version=torch.__version__,
        atol=1e-4,rtol=1e-4,command=[sys.executable,*sys.argv],rows=rows,
        max_abs_error_vs_pytorch_fp32=max(r['max_abs_errors']['fp32'] for r in rows),
        max_abs_error_vs_pytorch_fp64=max(r['max_abs_errors']['fp64'] for r in rows),
        report_sha256={f'compat-depth{d}.json':hashlib.sha256((out/f'compat-depth{d}.json').read_bytes()).hexdigest() for d in (1,2)})
    (out/'pytorch-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FP32_DEST_PYTORCH_ALL_PASS',flush=True)


if __name__=='__main__':main()
