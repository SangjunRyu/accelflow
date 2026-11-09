"""Copy the authorized Windows workspace to WSL and verify its entire manifest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

source = Path('/mnt/c/Users/ysjun5656/AccelFlow').resolve()
target = Path('/home/ysjun5656/dev/accelflow')
assert source.is_dir()
if target.exists():
    raise SystemExit('Target already exists; refusing to merge or overwrite it')

def manifest(root):
    files = {}
    directories = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise SystemExit(f'Symlink requires separate handling: {path}')
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files[relative] = {'size': path.stat().st_size,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            raise SystemExit(f'Unsupported file: {path}')
    return {'files': files, 'directories': directories}

before = manifest(source)
shutil.copytree(source, target, copy_function=shutil.copy2)
after = manifest(target)
assert before == after, 'Copied tree does not match the source'
assert before == manifest(source), 'Source changed while copying'
report = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'source_windows': 'C:\\Users\\ysjun5656\\AccelFlow',
    'target_linux': str(target),
    'target_windows': '\\\\wsl.localhost\\Ubuntu-24.04\\home\\ysjun5656\\dev\\accelflow',
    'file_count': len(before['files']), 'directory_count': len(before['directories']),
    'total_file_bytes': sum(item['size'] for item in before['files'].values()),
    'copy_verified': True, 'source_cleanup_completed': False, 'manifest': before
}
destination = target / 'results/setup/workspace-migration.json'
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'manifest'}, indent=2))
