"""Qwen3-VL through Apple's MLX runtime, behind the same private-worker protocol.

Measured on an M4 Air against the tinygrad path on the same image: 2B went
from ~36 s to ~6 s per description, and 8B runs in ~19 s. The worker is a
child process speaking JSON lines over pipes (never a port); it heartbeats
every 20 s so the parent's timeout measures silence, not slow hardware.
"""
import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

REPOS = {2: 'mlx-community/Qwen3-VL-2B-Instruct-4bit', 8: 'mlx-community/Qwen3-VL-8B-Instruct-4bit'}


def local_model_dir(size):
    """A bundled or downloaded snapshot; packaged builds never fetch implicitly."""
    name = REPOS[int(size)].split('/')[-1]
    roots = [os.environ.get('CLEARCAM_MODEL_DIR'), os.environ.get('CLEARCAM_DATA_DIR')]
    for root in roots:
        if not root: continue
        for candidate in (Path(root) / 'mlx' / name, Path(root) / 'models' / 'mlx' / name):
            if (candidate / 'config.json').is_file(): return candidate
    return None


def available_sizes():
    """Sizes whose weights are present locally (or fetchable in a dev checkout)."""
    if os.environ.get('CLEARCAM_NATIVE') == '1':
        return [size for size in REPOS if local_model_dir(size)]
    return list(REPOS)


DOWNLOAD_STATE = {'size': None, 'state': 'idle', 'bytes': 0, 'error': None}
APPROX_BYTES = {2: 1_800_000_000, 8: 5_400_000_000}


def download_dir(size):
    root = os.environ.get('CLEARCAM_DATA_DIR')
    if not root: raise RuntimeError('No data directory')
    return Path(root) / 'models' / 'mlx' / REPOS[int(size)].split('/')[-1]


def start_download(size):
    """Fetch a description model into the data directory. Explicit, never implicit."""
    import threading
    size = int(size)
    if size not in REPOS: raise ValueError('Unknown model size')
    if DOWNLOAD_STATE['state'] == 'downloading': raise RuntimeError('A download is already running')
    target = download_dir(size)
    DOWNLOAD_STATE.update(size=size, state='downloading', bytes=0, error=None)

    def run():
        try:
            from huggingface_hub import snapshot_download
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(target.name + '.partial')
            snapshot_download(REPOS[size], local_dir=str(staging))
            if target.exists():
                import shutil; shutil.rmtree(target)
            staging.replace(target)
            DOWNLOAD_STATE.update(state='ready', bytes=sum(f.stat().st_size for f in target.rglob('*') if f.is_file()))
        except Exception as error:
            DOWNLOAD_STATE.update(state='error', error=f'{type(error).__name__}: {error}')

    threading.Thread(target=run, daemon=True, name='MlxDownload').start()


def download_progress():
    state = dict(DOWNLOAD_STATE)
    if state['state'] == 'downloading' and state['size']:
        try:
            staging = download_dir(state['size']).with_name(download_dir(state['size']).name + '.partial')
            state['bytes'] = sum(f.stat().st_size for f in staging.rglob('*') if f.is_file()) if staging.exists() else 0
        except Exception:
            pass
        state['approx_total'] = APPROX_BYTES.get(state['size'])
    return state


def runtime_available():
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        return False


class MlxProcess:
    """Drop-in for QwenProcess: same generate()/close() surface."""

    def __init__(self, size='2B', res=(448, 448), timeout=300):
        self.size = int(str(size).rstrip('B'))
        self.timeout = timeout
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'utils.mlx_describer', '--worker'],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        atexit.register(self.close)

    def generate(self, *, image_path=None, prompt, reset=True, max_tokens=96, quiet=True, on_state=None):
        request = dict(size=self.size, image_path=str(image_path) if image_path is not None else None,
                       prompt=prompt, max_tokens=max_tokens)
        try:
            self.process.stdin.write((json.dumps(request) + '\n').encode())
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            absolute_deadline = time.monotonic() + 1800
            pending = b''
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = min(deadline, absolute_deadline) - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError('Local MLX worker exceeded its time limit')
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('Local MLX worker stopped unexpectedly')
                    deadline = time.monotonic() + self.timeout
                    pending += chunk
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        response = json.loads(line)
                        if 'error' in response: raise RuntimeError(response['error'])
                        if 'description' in response: return response['description']
                        if on_state and 'state' in response: on_state(response['state'])
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe and not pipe.closed: pipe.close()
        atexit.unregister(self.close)


def serve():
    import contextlib
    import threading
    import traceback
    real_stdout = sys.stdout
    lock = threading.Lock()

    def reply(payload):
        with lock:
            real_stdout.write(json.dumps(payload) + '\n')
            real_stdout.flush()

    loaded = {}
    for line in sys.stdin:
        stop = threading.Event()
        threading.Thread(target=lambda: [reply({'state': 'working'}) for _ in iter(lambda: not stop.wait(20), False)], daemon=True).start()
        try:
            request = json.loads(line)
            size = int(request['size'])
            if size not in loaded:
                reply({'state': 'loading_model'})
                with contextlib.redirect_stdout(sys.stderr):
                    from mlx_vlm import load
                    from mlx_vlm.utils import load_config
                    source = local_model_dir(size)
                    if source is None and os.environ.get('CLEARCAM_NATIVE') == '1':
                        raise RuntimeError('This description model is not included in this build.')
                    path = str(source) if source else REPOS[size]
                    model, processor = load(path)
                    loaded[size] = (model, processor, load_config(path))
            model, processor, config = loaded[size]
            reply({'state': 'describing'})
            with contextlib.redirect_stdout(sys.stderr):
                from mlx_vlm import generate
                from mlx_vlm.prompt_utils import apply_chat_template
                images = [request['image_path']] if request['image_path'] else []
                formatted = apply_chat_template(processor, config, request['prompt'], num_images=len(images))
                out = generate(model, processor, formatted, images, max_tokens=int(request['max_tokens']), verbose=False)
                text = (out.text if hasattr(out, 'text') else out) or ''
            stop.set()
            reply({'description': str(text).strip()})
        except Exception:
            stop.set()
            reply({'error': traceback.format_exc()})


if __name__ == '__main__':
    serve()
