"""One entry point for saved-evidence auditing and an optional TTSim MatMul smoke test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from portfolio_evidence import collect, require


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--evidence',action='store_true',help='Audit saved records only; requires Python standard library')
    mode.add_argument('--simulate',action='store_true',help='Audit records, build own host, run six small MatMul variants')
    args=parser.parse_args()
    expected=json.loads((root/'results/portfolio/20261110-public-summary/evidence.json').read_text())
    actual=collect(root)
    require(actual==expected,'Saved evidence differs from the curated public manifest')
    print(f'EVIDENCE_PASS sources={len(actual["source_sha256"])} models=7 cuda_rows=6 ttsim_rows=12',flush=True)
    if not args.simulate:
        print('Saved-record audit only: no kernels were executed.',flush=True)
        return

    metal=Path(os.environ.get('TT_METAL_HOME',Path.home()/'dev/tenstorrent/tt-metal-v0.79.0')).resolve()
    sim=Path(os.environ.get('TT_METAL_SIMULATOR',Path.home()/'dev/tenstorrent/sim/libttsim_wh.so')).resolve()
    require(sim.is_file(),'Set TT_METAL_SIMULATOR to the pinned Wormhole library; see docs/04 and docs/06')
    revision=subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip()
    digest=hashlib.sha256(sim.read_bytes()).hexdigest()
    require(revision=='de546d3b146758714d900f11b218c8f9c805f410','Expected TT-Metal v0.79.0')
    require(digest=='196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a','Expected TTSim v1.11.1 Wormhole')
    require((metal/'build_Release/tt_metal/libtt_metal.so').is_file(),
            'TT-Metal must already be built; this script does not install or build upstream dependencies')
    out=root/'results/portfolio'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    sources=[root/'tools/report/reproduce_portfolio.py',root/'tools/report/portfolio_evidence.py',
             root/'tools/report/build_matmul.py',root/'scripts/reproduce_portfolio.sh',root/'npu/vector_add/common.hpp']
    sources += [p for p in (root/'npu/matmul').rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.txt')]
    metadata=dict(passed=False,tt_metal_revision=revision,ttsim_sha256=digest,runs=[],
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        measurement_policy='Correctness smoke test; retained TTSim/host samples are not a repeat of published benchmarks or real-chip performance')
    for path in sources:
        target=out/'sources'/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    def save():(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    save();print('RESULTS='+str(out),flush=True)
    env=dict(os.environ,TT_METAL_HOME=str(metal),TT_METAL_RUNTIME_ROOT=str(metal),
             TT_METAL_SIMULATOR=str(sim),TT_METAL_SLOW_DISPATCH_MODE='1')
    with (out/'host-build.log').open('w') as log:
        subprocess.run(['python3',str(root/'tools/report/build_matmul.py')],cwd=root,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=root/'build/matmul/matmul'
    metadata['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    shutil.copyfile(metal/'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml',sim.parent/'soc_descriptor.yaml')
    fingerprints=[]
    for variant in ('baseline','reuse_a','multicore','mcast','mcast2d','mcast2d_batch4'):
        cores=4 if variant not in ('baseline','reuse_a') else 1
        # Existing host selector: five samples + two warmups of one small integer fixture.
        command=[str(exe),str(out/(variant+'.json')),variant,'2',str(cores),
                 '--benchmark','5','signed_random:64,64,96']
        start=time.perf_counter()
        with (out/(variant+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        metadata['runs'].append(dict(variant=variant,command=command,exit_code=result.returncode,
            process_wall_seconds=time.perf_counter()-start))
        save();require(result.returncode==0,f'Failed smoke test; inspect {out/variant}.log')
        data=json.loads((out/(variant+'.json')).read_text())
        require(data['passed'] and len(data['cases'])==1,variant)
        case=data['cases'][0]
        require(case['passed'] and case['max_abs_error']==0 and case['padding_verified'] and
                case['output_sentinel_overwritten'] and case['layout_matches_official_tilizer'] and
                case['timing']['all_repetition_outputs_equal'],variant)
        require(case['shape_mkn']==[64,64,96] and case['timing']['sample_count']==5 and
                case['timing']['warmup_count']==2,variant)
        fingerprints.append([case[k] for k in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint')])
        require(all(f==fingerprints[0] for f in fingerprints),'Inputs or outputs differ across smoke variants')
        print('SIMULATOR_PASS '+variant,flush=True)
    metadata.update(passed=True,variants=6,operator_submissions=42,all_fingerprints_equal=True)
    save();print('PORTFOLIO_SMOKE_ALL_PASS '+str(out),flush=True)


if __name__=='__main__':main()
