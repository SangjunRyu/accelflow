"""Export actual host and SFPI JIT artifacts for learning; no TT-MLIR graph is inferred."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

root = Path(__file__).resolve().parents[2]
metal = Path.home() / 'dev/tenstorrent/tt-metal-v0.79.0'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=('streamed', 'blocked', 'baseline'), default='streamed')
args = parser.parse_args()
suffix = '_' + args.variant if args.variant != 'baseline' else ''
names = ['reader' + suffix, 'compute' + suffix,
         'writer_blocked' if suffix else 'writer']
cache = Path.home() / '.cache/tt-metal-cache'
out = root / 'results/compile_inspection' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'variant': args.variant,
            'tt_metal_revision': subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'],text=True).strip(),
            'scope': 'actual host compilation and existing device JIT cache, not TT-MLIR IR',
            'artifact_selection': 'all matching kernel-name folders with ELF files; hash variants retained explicitly',
            'device_artifacts': [], 'passed': False}
if manifest['tt_metal_revision'] != 'de546d3b146758714d900f11b218c8f9c805f410':
    raise SystemExit('Expected pinned TT-Metal v0.79.0')
bin_dir = metal / 'runtime/sfpi/compiler/bin'
tools = {name: bin_dir / ('riscv-tt-elf-' + name) for name in ('objdump','readelf','nm','size')}
for tool in tools.values():
    if not tool.is_file():
        raise SystemExit(f'Missing installed SFPI tool: {tool}')

def capture(command, destination):
    with destination.open('w') as log:
        subprocess.run([str(part) for part in command], stdout=log, stderr=subprocess.STDOUT, check=True)

try:
    # 검증된 host build interface를 재사용하고 upstream 설정은 변경하지 않습니다.
    host = out / 'host'
    host.mkdir()
    build = root / 'build/matmul'
    interface = build / 'metal-interface.cmake'
    if not interface.is_file():
        raise SystemExit('Run scripts/run_matmul.sh once to prepare the verified host interface')
    capture(['cmake','-S',root/'npu/matmul','-B',build,'-G','Ninja',
             '-DCMAKE_CXX_COMPILER=clang++-20',f'-DMETAL_INTERFACE={interface}',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'], host/'configure.log')
    database = json.loads((build/'compile_commands.json').read_text())
    shutil.copyfile(build/'compile_commands.json', host/'compile_commands.json')
    entry = next(row for row in database if Path(row['file']).name == 'matmul.cpp')
    command = shlex.split(entry['command'])
    filtered = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in ('-o','-MF','-MT'):
            index += 2
            continue
        if token not in ('-MD','-MMD','-c'):
            filtered.append(token)
        index += 1
    ir_command = filtered + ['-S','-emit-llvm','-o',str(host/'matmul-host.ll')]
    manifest['host_IR_command'] = ir_command
    capture(ir_command, host/'llvm-ir.log')
    assert (host/'matmul-host.ll').stat().st_size > 0

    # 기존 cache는 읽기 전용으로 다루며 ELF와 생성 wrapper를 hash별로 복사합니다.
    folders = []
    for name in names:
        folders += [p for p in cache.glob(f'*/kernels/{name}/*')
                    if p.is_dir() and any(p.glob('*/*.elf'))]
    if not folders:
        raise SystemExit('No matching JIT artifacts; run the corresponding simulator example first')
    total_elf_bytes = sum(p.stat().st_size for folder in folders for p in folder.glob('*/*.elf'))
    if total_elf_bytes > 20 * 1024 * 1024:
        raise SystemExit('ELF copy estimate exceeds 20MiB; narrow the cache selection before exporting')
    manifest['elf_copy_estimate_bytes'] = total_elf_bytes
    for folder in sorted(folders):
        name, key = folder.parent.name, folder.name
        destination = out / 'device' / folder.parents[2].name / name / key
        destination.mkdir(parents=True)
        for source in folder.iterdir():
            if source.is_file() and source.suffix in ('.cpp','.h'):
                shutil.copyfile(source, destination/source.name)
        for elf in sorted(folder.glob('*/*.elf')):
            if '.xip.' in elf.name or 'weakened' in elf.name:
                continue
            processor = elf.parent.name
            target = destination / processor
            target.mkdir(exist_ok=True)
            shutil.copyfile(elf, target/elf.name)
            for extension in ('.log','.dephash'):
                companion = Path(str(elf) + extension)
                if companion.is_file():
                    shutil.copyfile(companion, target/companion.name)
            commands = {
                'assembly.txt': [tools['objdump'],'-d','-C',elf],
                'elf-info.txt': [tools['readelf'],'-h','-S','-s',elf],
                'symbols.txt': [tools['nm'],'-C','--defined-only',elf],
                'sections.txt': [tools['size'],'-A',elf],
            }
            for output_name, command in commands.items():
                capture(command, target/output_name)
            manifest['device_artifacts'].append({
                'kernel': name, 'hash_variant': key, 'processor': processor,
                'source_elf': str(elf), 'copied_elf': str((target/elf.name).relative_to(out)),
                'bytes': elf.stat().st_size, 'sha256': hashlib.sha256(elf.read_bytes()).hexdigest(),
                'inspection_commands': {k:[str(x) for x in v] for k,v in commands.items()},
            })
    manifest['passed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Evidence:', out, flush=True)

lines = ['# Compilation inspection artifacts', '',
         'Host LLVM IR describes the x86_64 host program; it is not TTIR/TTNN or Tensix IR.',
         'Device files are real SFPI-produced ELF/cache artifacts; cached hash variants may use different configs.', '',
         '[Manifest](manifest.json) · [Host compile commands](host/compile_commands.json) · [Host LLVM IR](host/matmul-host.ll)', '',
         '| Kernel | Cache hash | Processor | Bytes | Disassembly | ELF metadata |',
         '| --- | --- | --- | --- | --- | --- |']
for row in manifest['device_artifacts']:
    parent = Path(row['copied_elf']).parent.as_posix()
    lines.append(f'| {row["kernel"]} | {row["hash_variant"]} | {row["processor"]} | {row["bytes"]} | '
                 f'[assembly]({parent}/assembly.txt) | [ELF]({parent}/elf-info.txt) |')
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(f'PASS: host LLVM IR + {len(manifest["device_artifacts"])} device ELF inspections')
