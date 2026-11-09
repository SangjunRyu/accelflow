"""Check transport reconstruction against actual fixture FP32 squares and statistics."""
import argparse,hashlib,json,math,struct
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
manifest=json.loads((out/'fixtures.json').read_text())
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def word(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def value(w):return struct.unpack('<f',struct.pack('<I',w))[0]
def remainder_bits(original,r):
    if not r:return 0
    e=(original>>23)&255;leading=r.bit_length()-1;new=e-23+leading
    return ((new<<23)|((r<<(23-leading))&0x7fffff)) if new>0 else r<<(e-1 if e else 0)
count=0;statistics=0
for f in manifest['fixtures']:
    if f['performance']:continue
    data=(out/f['binary']).read_bytes();_,h,w,hp,wp,eps=struct.unpack_from('<8sIIIIf',data)
    bits=struct.unpack_from('<'+'H'*(hp*wp+wp),data,28)
    for row in range(h):
        sums=0.0
        for col in range(w):
            x=value(bits[row*wp+col]<<16);square=f32(x*x);v=word(square)
            hi=v&0xffffe000;lo=remainder_bits(v,v&0x1fff)
            assert hi&0x1fff==lo&0x1fff==0
            assert f32(value(hi)+value(lo))==square
            sums=f32(sums+square);count+=1
        v=word(sums);parts=[v&0xffffe000,remainder_bits(v,v&0x1ff8),remainder_bits(v,v&7)]
        assert all(p&0x1fff==0 for p in parts)
        assert f32(f32(value(parts[0])+value(parts[1]))+value(parts[2]))==sums
        statistics+=1
result=dict(passed=True,square_values_checked=count,row_statistics_checked=statistics,
            property='Every transmitted component is TF32-representable; FP32 reconstruction equals original bits for this corpus',
            scope='actual finite BF16 fixtures; not all FP32/subnormal/NaN/Inf values')
(out/'precision-transport-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PRECISION_TRANSPORT_PASS',count,statistics)
