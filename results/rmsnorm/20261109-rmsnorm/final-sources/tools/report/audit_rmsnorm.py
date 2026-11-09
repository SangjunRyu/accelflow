"""Independently verify BF16 inputs, Python FP64 outputs and cross-backend RMSNorm results."""
import argparse
from array import array
import hashlib,json,math,struct,sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output
manifest=json.loads((out/'fixtures.json').read_text());assert manifest['passed']
npu=json.loads((out/'npu-depth1.json').read_text());assert npu['passed']
npu_cases={c['key']:c for c in npu['cases']}
measurements=json.loads((out/'cuda-measurements.json').read_text())

def float_bits(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]

def packed_fingerprint(bits,h,w):
    value=14695981039346656037
    for tr in range(0,h,32):
        for tc in range(0,w,32):
            for fr in range(2):
                for fc in range(2):
                    for rr in range(16):
                        for cc in range(16):
                            word=bits[(tr+fr*16+rr)*w+tc+fc*16+cc]<<16
                            for byte in struct.pack('<I',word):value=((value^byte)*1099511628211)&((1<<64)-1)
    return value

checked=[];max_cross=0;max_python=0
for fixture in manifest['fixtures']:
    binary=out/fixture['binary'];data=binary.read_bytes()
    assert hashlib.sha256(data).hexdigest()==fixture['sha256']
    if fixture['performance']:continue
    magic,h,w,hp,wp,epsilon=struct.unpack_from('<8sIIIIf',data)
    assert magic==b'AFRM0001' and len(data)==28+(hp*wp+wp)*2
    bits=array('H');bits.frombytes(data[28:])
    if sys.byteorder!='little':bits.byteswap()
    xbits=bits[:hp*wp];gbits=bits[hp*wp:]
    case=npu_cases[fixture['key']]
    assert packed_fingerprint(xbits,hp,wp)==case['input_fingerprint']
    assert packed_fingerprint(gbits*32,32,wp)==case['gamma_fingerprint']
    reference=[]
    for row in range(h):
        values=[float_bits(xbits[row*wp+col]) for col in range(w)]
        inverse=1/math.sqrt(math.fsum(v*v for v in values)/w+epsilon)
        reference.extend(v*inverse*float_bits(gbits[col]) for col,v in enumerate(values))
    variants={}
    for variant in ['unfused','fused']:
        m=next(m for m in measurements if m['key']==fixture['key'] and m['variant']==variant)
        variants[variant]=json.loads((out/m['report']).read_text())['logical_output']
    outputs=[case['logical_output'],variants['unfused'],variants['fused']]
    for values in outputs:
        assert len(values)==h*w
        for got,expected in zip(values,reference,strict=True):
            error=abs(got-expected)
            assert math.isfinite(got) and error<=1e-4+1e-4*abs(expected),(fixture['key'],got,expected)
            max_python=max(max_python,error)
    for i,expected in enumerate(reference):
        difference=max(v[i] for v in outputs)-min(v[i] for v in outputs)
        assert difference<=1e-4+1e-4*abs(expected)
        max_cross=max(max_cross,difference)
    if fixture['key'].startswith(('zero_','gamma_zero_')):assert all(v==0 for values in outputs for v in values)
    checked.append(fixture['key'])
assert len(checked)==20
summary=dict(passed=True,shared_npu_cuda_cases=len(checked),input_fingerprints_equal=True,
             independent_reference='Python math.fsum and math.sqrt over actual BF16 input/gamma and serialized FP32 epsilon',
             atol=1e-4,rtol=1e-4,max_cross_backend_abs_difference=max_cross,
             max_python_reference_abs_error=max_python,keys=checked,
             limitations=['Finite BF16 test inputs only','No real NPU performance measurement','No bank-conflict hardware counters'])
(out/'cross-backend-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print('RMSNORM_CROSS_BACKEND_PASS cases=20 max_difference='+str(max_cross),flush=True)
