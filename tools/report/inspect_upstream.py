"""Read official metadata/source snapshots without cloning or building upstream."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

out = Path('results/setup/upstream-2026-10-06')
out.mkdir(parents=True, exist_ok=True)

def fetch(url, name):
    with urlopen(Request(url, headers={'User-Agent': 'AccelFlow-Phase0'}), timeout=40) as r:
        data = r.read()
    (out / name).write_bytes(data)
    return data

for repo in ('ttsim', 'tt-metal'):
    data = json.loads(fetch(f'https://api.github.com/repos/tenstorrent/{repo}/releases/latest', f'{repo}-release.json'))
    print(repo, 'latest', data['tag_name'], data['target_commitish'])
    if repo == 'ttsim':
        for asset in data['assets']:
            if asset['name'] == 'libttsim_wh.so':
                print('ASSET', json.dumps({k: asset.get(k) for k in ('name','size','digest','browser_download_url')}))

for ref in ('main', 'v0.78.0', 'v0.79.0'):
    for path in ('INSTALLING.md', 'install_dependencies.sh', 'build_metal.sh', 'CMakeLists.txt', '.gitmodules',
                 'cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake', 'tt_metal/sfpi-info.sh'):
        fetch(f'https://raw.githubusercontent.com/tenstorrent/tt-metal/{ref}/{path}', 'metal-' + ref + '-' + path.replace('/', '_'))

for path in ('README.md','docs/libttsim_api.md'):
    fetch(f'https://raw.githubusercontent.com/tenstorrent/ttsim/v1.11.1/{path}', 'sim-' + path.replace('/', '_'))
for repo, ref in (('tt-metal','v0.79.0'), ('ttsim','v1.11.1')):
    commit = json.loads(fetch(f'https://api.github.com/repos/tenstorrent/{repo}/commits/{ref}', f'{repo}-commit.json'))
    print(repo, 'COMMIT', commit['sha'])
    if repo == 'tt-metal':
        tree = json.loads(fetch(f'https://api.github.com/repos/tenstorrent/{repo}/git/trees/{commit["sha"]}?recursive=1', 'metal-tree.json'))
        print('TREE truncated', tree.get('truncated'))
        for item in tree['tree']:
            if item['type'] == 'commit' or 'add_2_integers' in item['path'] or 'simulat' in item['path'].lower():
                print('TREE', item['path'], item.get('size'), item['sha'])
        print('MAIN_TREE_FILE_BYTES', sum(i.get('size',0) for i in tree['tree'] if i['type']=='blob'))
        for path in ('tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp',
                     'tt_metal/programming_examples/add_2_integers_in_riscv/CMakeLists.txt',
                     'tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml'):
            fetch(f'https://raw.githubusercontent.com/tenstorrent/tt-metal/{ref}/{path}', 'metal-' + path.replace('/', '_'))
        umd = next(i for i in tree['tree'] if i['path']=='tt_metal/third_party/umd')
        u_tree = json.loads(fetch(f'https://api.github.com/repos/tenstorrent/tt-umd/git/trees/{umd["sha"]}?recursive=1', 'umd-tree.json'))
        print('UMD_FILE_BYTES', sum(i.get('size',0) for i in u_tree['tree'] if i['type']=='blob'))
        for i in u_tree['tree']:
            if i['type']=='blob' and ('simulat' in i['path'].lower() or 'ttsim' in i['path'].lower()):
                print('UMD SOURCE', i['path'])
                fetch(f'https://raw.githubusercontent.com/tenstorrent/tt-umd/{umd["sha"]}/{i["path"]}', 'umd-' + i['path'].replace('/', '_'))
    else:
        tree = json.loads(fetch(f'https://api.github.com/repos/tenstorrent/{repo}/git/trees/{commit["sha"]}?recursive=1', 'sim-tree.json'))
        for i in tree['tree']:
            if i['type']=='blob' and 'libttsim' in i['path'] and i['path'].endswith(('.cc','.cpp','.h')):
                print('SIM SOURCE', i['path'])
                fetch(f'https://raw.githubusercontent.com/tenstorrent/ttsim/{ref}/{i["path"]}', 'sim-' + i['path'].replace('/', '_'))
