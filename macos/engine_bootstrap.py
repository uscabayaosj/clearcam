"""Native-only entrypoint: one owner per data directory and parent supervision."""
import fcntl
import os
from pathlib import Path
import runpy
import signal
import sys
import threading
import time

if __name__ == '__main__':
    os.umask(0o077)
    try:
        os.setsid()
    except PermissionError:
        pass  # already a group leader when launched directly by the app
    data = Path(os.environ['CLEARCAM_DATA_DIR'])
    data.mkdir(parents=True, exist_ok=True)
    lock = (data / 'engine.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('ClearCam already owns this data folder. Quit the other instance first.')
    parent = int(os.environ['CLEARCAM_PARENT_PID'])
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    def watch_parent():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=watch_parent, daemon=True).start()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / 'clearcam.py'), run_name='__main__')
