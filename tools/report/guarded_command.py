"""Run a command with a 10 GiB host-disk reserve; stream output and log it."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

log_path, *command = sys.argv[1:]
assert command
reserve = 10 * 1024**3
if shutil.disk_usage('/mnt/c').free < reserve:
    raise SystemExit('C: free space is below the 10 GiB reserve')
with Path(log_path).open('a') as log:
    log.write('Command: ' + repr(command) + '\n')
    log.flush()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    def monitor():
        while process.poll() is None:
            if shutil.disk_usage('/mnt/c').free < reserve:
                print('Stopping command: C: free space below 10 GiB', flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(5)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                return
            time.sleep(5)
    threading.Thread(target=monitor, daemon=True).start()
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    code = process.wait()
    log.write(f'command_exit_code={code}\n')
raise SystemExit(code)
