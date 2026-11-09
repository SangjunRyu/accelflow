"""Record verified bootstrap results and the actual build environment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root = Path(__file__).resolve().parents[2]
checkout = Path.home() / 'dev/tenstorrent/tt-metal-v0.79.0'
simulator = Path.home() / 'dev/tenstorrent/sim/libttsim_wh.so'
logs = sorted((root / 'results/setup').glob('metal-smoke-*.log'))
if not logs:
    raise SystemExit('No smoke-test log exists')
log = logs[-1]
output = log.read_text()
assert 'example_exit_code=0' in output and 'Success: Result is 21' in output
def capture(*command):
    return subprocess.check_output(command, text=True).strip()
report = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'backend': 'ttsim', 'architecture': 'wormhole',
    'operator': 'official add_2_integers_in_riscv', 'expected_integer_result': 21,
    'example_exit_code': 0, 'output_verified': True,
    'smoke_log': str(log.relative_to(root)),
    'tt_metal_tag': 'v0.79.0', 'tt_metal_commit': capture('git','-C',str(checkout),'rev-parse','HEAD'),
    'submodules': capture('git','-C',str(checkout),'submodule','status','--recursive'),
    'upstream_checkout_changes': capture('git','-C',str(checkout),'status','--short'),
    'ttsim_tag': 'v1.11.1', 'ttsim_sha256': hashlib.sha256(simulator.read_bytes()).hexdigest(),
    'host_compiler': capture('clang++-20','--version'),
    'gcc14': capture('g++-14','--version'), 'cmake': capture('cmake','--version'),
    'ninja': capture('ninja','--version'), 'python': capture('python3','--version'),
    'os_release': Path('/etc/os-release').read_text(), 'kernel': capture('uname','-r'),
    'memory': capture('free','-b'), 'host_backing_volume_free_bytes': shutil.disk_usage('/mnt/c').free,
    'checkout_disk_usage': capture('du','-sh',str(checkout)),
    'build_options': 'Release; 2 jobs; upstream Unity ON; PCH/Tracy/MPI/Python bindings OFF; selected example target',
    'timing_policy': 'No simulator wall-clock performance claim; functional smoke test only'
}
destination = root / 'results/setup/tt-metal-bootstrap-metadata.json'
destination.write_text(json.dumps(report, indent=2)+'\n')
print(destination)
