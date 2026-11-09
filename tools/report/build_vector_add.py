"""Reuse the pinned build tree's verified host interface without rebuilding TT-Metal."""
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[2]
metal = Path(os.environ.get('TT_METAL_HOME', str(Path.home()/'dev/tenstorrent/tt-metal-v0.79.0'))).resolve()
build = metal / 'build_Release'
revision = subprocess.check_output(['git','-C',str(metal),'rev-parse','HEAD'], text=True).strip()
if revision != 'de546d3b146758714d900f11b218c8f9c805f410':
    raise SystemExit('This build-tree adapter is validated only for TT-Metal v0.79.0')
commands = subprocess.check_output(['ninja','-C',str(build),'-t','commands',
                                   'metal_example_add_2_integers_in_riscv'], text=True).splitlines()
compile_tokens = next(shlex.split(line) for line in commands
    if ' -c ' in line and line.endswith('/add_2_integers_in_riscv.cpp'))
includes, defines = [], []
tokens = iter(compile_tokens)
for token in tokens:
    if token == '-isystem': includes.append(next(tokens))
    elif token.startswith('-I'): includes.append(token[2:])
    elif token.startswith('-D') and not token.startswith('-DOVERRIDE_KERNEL_PREFIX'): defines.append(token[2:])
libs = [build/'tt_metal/libtt_metal.so', build/'tt_stl/libtt_stl.so',
        build/'_deps/spdlog-build/libspdlog.a', build/'lib/libtracy.so',
        build/'tt_metal/third_party/umd/lib/libtt-umd.so']
for library in libs:
    if not library.is_file(): raise SystemExit(f'Missing built library: {library}')
out = root/'build/vector_add'
out.mkdir(parents=True, exist_ok=True)
interface = out/'metal-interface.cmake'
def cmake_list(name, values):
    return f'set({name}\n' + ''.join(f'  [==[{v}]==]\n' for v in values) + ')\n'
interface.write_text(cmake_list('METAL_INCLUDES', includes) + cmake_list('METAL_DEFINES', defines)
    + cmake_list('METAL_LINK_LIBRARIES', map(str,libs))
    + cmake_list('METAL_LIBRARY_DIRS', sorted({str(p.parent) for p in libs})))
(out/'metal-interface.json').write_text(json.dumps({'revision':revision,'includes':includes,'defines':defines},indent=2))
subprocess.run(['cmake','-S',str(root/'npu/vector_add'),'-B',str(out),'-G','Ninja',
                '-DCMAKE_CXX_COMPILER=clang++-20',f'-DMETAL_INTERFACE={interface}'],check=True)
subprocess.run(['cmake','--build',str(out),'--parallel','2'],check=True)
