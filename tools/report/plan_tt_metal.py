"""Measure source bytes and apt's proposal; do not install, clone, or build."""
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

out = Path('results/setup/upstream-2026-10-06')
def fetch(url, name):
    with urlopen(Request(url, headers={'User-Agent': 'AccelFlow-Phase0'}), timeout=40) as r:
        data = r.read()
    (out / name).write_bytes(data)
    return data

sizes = {'tt-metal': 405201675, 'tt-umd': 4416834}
for repo, commit, name in (
    ('tenstorrent-metal/tracy', '9c3ca7e5fe48cecf90aa518c82552aae706cf110', 'tracy'),
    ('tenstorrent/tt-cluster-descriptors', 'ac8a3b5ba3a54ad4e93cb75da3692942e5926c50', 'cluster')):
    tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1', f'{name}-tree.json'))
    assert not tree.get('truncated')
    assert not any(i['type']=='commit' for i in tree['tree']), 'Nested submodule needs accounting'
    sizes[name] = sum(i.get('size',0) for i in tree['tree'] if i['type']=='blob')
for path in ('tt_metal/sfpi-version','tt_metal/programming_examples/CMakeLists.txt', 'cmake/dependencies.cmake'):
    try:
        fetch(f'https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.79.0/{path}', 'metal-' + path.replace('/', '_'))
    except Exception as error:
        print('Optional source', path, error)
proposal = ['clang-20','lld-20','g++-14','libc++-20-dev','libc++abi-20-dev','libatomic1',
            'libstdc++6','xz-utils','pandoc','python3-pip','python3-pkg-resources','xxd']
result = subprocess.run(['apt-get','--assume-no','--no-install-recommends','install',*proposal],
                        capture_output=True, text=True, env={**__import__('os').environ,'LC_ALL':'C'})
Path('results/setup/tt-metal-toolchain-proposal-2026-10-06.log').write_text(result.stdout + result.stderr)
print(result.stdout + result.stderr)
assert result.returncode == 1 and 'Abort.' in result.stdout, 'Expected an explicit non-installing apt refusal'
budget = {
    'source_blob_bytes': sizes, 'total_source_blob_bytes': sum(sizes.values()),
    'sources_are_uncompressed_git_blobs_not_clone_disk_measurements': True,
    'toolchain_proposal_packages': proposal,
    'additional_disk_budget_gib': {
        'shallow_sources_git_and_filesystem_overhead': [1, 2],
        'host_toolchains_and_download_cache': [2, 3],
        'sfpi_and_cmake_fetched_dependencies': [2, 4],
        'build_objects_pch_libraries_runtime_cache': [10, 25]
    },
    'additional_disk_budget_total_gib': [15,34],
    'host_free_reserve_gib': 10,
    'conservative_required_free_gib': 44,
    'build_estimate_is_planning_allowance_not_measured_or_upstream_minimum': True,
    'build_decision': 'defer; 33.26 GiB current C: free is below conservative 44 GiB allowance'
}
Path('results/setup/tt-metal-resource-budget-2026-10-06.json').write_text(json.dumps(budget, indent=2)+'\n')
print(json.dumps(budget, indent=2))
