"""Create shared BF16 RMSNorm inputs and a Python FP64 reference description."""
import argparse
from pathlib import Path
import hashlib
import json
import random
import struct

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args(); out=a.output; scratch=out/'scratch'; scratch.mkdir(parents=True,exist_ok=True)

def bf16(x):
    word=struct.unpack('<I',struct.pack('<f',x))[0]
    return ((word+0x7fff+((word>>16)&1))>>16)&65535

cases=[('deterministic',2,3,1e-5),('zero',33,35,1e-5),('ones',1,1,1e-5),
       ('ones',2,31,1e-5),('ones',2,32,1e-5),('ones',2,33,1e-5),
       ('signed',31,65,1e-5),('signed',32,65,1e-5),('signed',33,65,1e-5),
       ('signed',65,129,1e-5),('signed',3,257,1e-5),('signed',2,1025,1e-5),
       ('signed',1,4097,1e-5),('fractional',17,257,1e-5),('tiny',3,33,1e-5),
       ('mixed',3,65,1e-5),('gamma_zero',3,35,1e-5),('gamma_signed',33,65,1e-5),
       ('epsilon',3,33,1e-3),('epsilon',3,33,1e-6)]
fixtures=[]
for performance, entries in [(False,cases),(True,[('perf',128,320,1e-5),('perf',640,320,1e-5),('perf',320,4097,1e-5),('perf',1024,8193,1e-5)])]:
    for name,h,w,eps in entries:
        hp,wp=(h+31)//32*32,(w+31)//32*32
        key=f'{name}_{h}x{w}_eps{eps:g}'
        rng=random.Random(2026+h*10000+w)
        values=[]
        for row in range(hp):
            for col in range(wp):
                v=0.0
                if row<h and col<w:
                    v=float((row*13+col*7)%17-8)
                    if name=='deterministic': v=float(row*w+col+1)
                    if name=='zero': v=0.0
                    if name=='ones': v=1.0
                    if name=='fractional': v=rng.uniform(-0.5,0.5)
                    if name=='tiny' or name=='epsilon': v=0.0001*(-1 if col%2 else 1)
                    if name=='mixed': v=[0.001,1.,32.,-32.,-0.001][col%5]
                values.append(bf16(v))
        gamma=[]
        for col in range(wp):
            v=1.0 if col<w else 0.0
            if col<w and name not in ['ones','zero','deterministic']:
                v=(col%9-4)/4.0
            if name=='gamma_zero': v=0.0
            gamma.append(bf16(v))
        binary=scratch/(key+'.bin')
        with binary.open('wb') as f:
            f.write(struct.pack('<8sIIIIf',b'AFRM0001',h,w,hp,wp,eps))
            f.write(struct.pack('<'+'H'*len(values),*values))
            f.write(struct.pack('<'+'H'*len(gamma),*gamma))
        fixtures.append(dict(key=key,shape_hw=[h,w],epsilon=struct.unpack('<f',struct.pack('<f',eps))[0],
                             performance=performance,binary='scratch/'+binary.name,
                             sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
manifest=dict(passed=True,seed=2026,input_dtype='BF16 values; expanded FP32 storage on both backends',
              output_dtype='FP32',atol=1e-4,rtol=1e-4,formula='x * rsqrt(sum(x*x)/W + epsilon) * gamma',fixtures=fixtures)
(out/'fixtures.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'npu-fixtures.txt').write_text(''.join(str((out/f['binary']).resolve())+'\n' for f in fixtures if not f['performance']))
(out/'pilot-fixtures.txt').write_text(''.join(str((out/f['binary']).resolve())+'\n' for f in fixtures[:2]))
print('RMSNORM_FIXTURES_PASS '+str(out),flush=True)
