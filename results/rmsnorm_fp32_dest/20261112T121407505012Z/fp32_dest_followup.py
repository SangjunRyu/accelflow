"""Extra precision controls for the saved FP32-to-Dest experiment."""
from pathlib import Path
import hashlib,json,math,os,struct,subprocess,sys

root=Path('/home/ysjun5656/dev/accelflow');out=Path(sys.argv[1]).resolve()
assert json.loads((out/'summary.json').read_text())['passed']
manifest=json.loads((out/'fixtures.json').read_text())
selected=[f for f in manifest['fixtures'] if f['key'].startswith(('fp32_lowbits_','fp32_fractional_'))]
paths=out/'default-control-fixtures.txt'
paths.write_text(''.join(str(out/f['binary'])+'\n' for f in selected))
metal=Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'
env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
    TT_METAL_SIMULATOR=str(metal.parent/'sim/libttsim_wh.so'),TT_METAL_SLOW_DISPATCH_MODE='1')
command=[str(root/'build/rmsnorm_fp32_dest/rmsnorm_fp32_dest'),'run',str(paths),
    str(out/'default-rmsnorm-control.json'),'1','1','default']
with (out/'default-rmsnorm-control.log').open('w') as log:
    process=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
control=json.loads((out/'default-rmsnorm-control.json').read_text())
assert process.returncode==1 and not control['passed']
assert all(not c['passed'] and c['bad_output_values']>0 for c in control['cases'])
statistics=[]
for depth in (1,2):
    data=json.loads((out/f'compat-depth{depth}.json').read_text())
    bykey={c['key']:c for c in data['cases']}
    for f in manifest['fixtures']:
        raw=(out/f['binary']).read_bytes()
        magic,h,w,hp,wp,eps=struct.unpack_from('<8sIIIIf',raw)
        if magic==b'AFRM0001':
            words=struct.unpack('<'+'H'*((len(raw)-28)//2),raw[28:])
            x=[struct.unpack('<f',struct.pack('<I',b<<16))[0] for b in words[:hp*wp]]
        else:x=struct.unpack_from('<'+'f'*(hp*wp),raw,28)
        case=bykey[f['key']]
        errors=[];ulps=[]
        for row,got in enumerate(case['logical_sum_squares']):
            reference=math.fsum(float(x[row*wp+c])**2 for c in range(w))
            errors.append(abs(got-reference)/max(abs(reference),1e-30))
            expected_bits=struct.unpack('<I',struct.pack('<f',reference))[0]
            actual_bits=struct.unpack('<I',struct.pack('<f',got))[0]
            ulps.append(abs(actual_bits-expected_bits))
        statistics.append(dict(key=f['key'],depth=depth,fixture_dtype=f['fixture_dtype'],
            max_relative_sum_error=max(errors),max_ulp_distance_from_rounded_fp64_sum=max(ulps)))
record=dict(passed=True,default_rmsnorm_negative_control_rejected=True,
    default_control_command=command,default_control_exit_code=process.returncode,
    default_control_max_abs_error=max(c['max_abs_error'] for c in control['cases']),
    default_control_bad_output_values=sum(c['bad_output_values'] for c in control['cases']),
    max_relative_sum_error=max(r['max_relative_sum_error'] for r in statistics),
    max_sum_ulp_distance=max(r['max_ulp_distance_from_rounded_fp64_sum'] for r in statistics),
    statistics=statistics,scope='Corpus metrics, not an all-input error bound or correctly-rounded reduction guarantee')
(out/'precision-controls.json').write_text(json.dumps(record,indent=2)+'\n')
(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print('FP32_PRECISION_CONTROLS_PASS',flush=True)
