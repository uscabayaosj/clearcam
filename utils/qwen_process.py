"""Keep tinygrad's thread-bound SQLite cache and GPU runtime out of the NVR.

The private child speaks JSON lines over inherited pipes, never a network port.
Model weights stay loaded between events. The parent bounds requests and kills
the child on shutdown or a timeout; diagnostics never contain camera credentials.
"""
import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time


class QwenProcess:
    def __init__(self, size='2B', res=(448, 448), timeout=300):
        self.size, self.res, self.timeout = size, res, timeout
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'utils.qwen_process', '--worker'],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        atexit.register(self.close)

    def generate(self, *, image_path=None, prompt, reset=True, max_tokens=96, quiet=True, on_state=None):
        request = dict(size=self.size, res=self.res,
                       image_path=str(image_path) if image_path is not None else None,
                       prompt=prompt, reset=reset, max_tokens=max_tokens)
        try:
            self.process.stdin.write((json.dumps(request) + '\n').encode())
            self.process.stdin.flush()
            # Inactivity timeout: the worker heartbeats every 20s while loading
            # or generating, so slow hardware is fine; only true silence kills.
            deadline = time.monotonic() + self.timeout
            absolute_deadline = time.monotonic() + 1800
            pending = b''
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = min(deadline, absolute_deadline) - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError('Local Qwen worker exceeded its time limit')
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('Local Qwen worker stopped unexpectedly')
                    deadline = time.monotonic() + self.timeout
                    pending += chunk
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        response = json.loads(line)
                        if 'error' in response:
                            raise RuntimeError(response['error'])
                        if 'description' in response:
                            return response['description']
                        if on_state and 'state' in response:
                            on_state(response['state'])
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
            if pipe and not pipe.closed:
                pipe.close()
        atexit.unregister(self.close)


def serve():
    import contextlib
    import threading
    import traceback
    model = None
    model_key = None
    real_stdout = sys.stdout  # redirect_stdout blocks must not swallow protocol lines
    reply_lock = threading.Lock()

    def reply(payload):
        with reply_lock:
            real_stdout.write(json.dumps(payload) + '\n')
            real_stdout.flush()

    for line in sys.stdin:
        heartbeat_stop = threading.Event()

        def heartbeat():
            # Model load and generation can be silent for minutes on slower
            # Macs; periodic liveness keeps the parent's inactivity timeout fed.
            while not heartbeat_stop.wait(20):
                reply({'state': 'working'})

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            request = json.loads(line)
            key = (request['size'], tuple(request['res']))
            if key != model_key:
                reply({'state': 'loading_model'})
                with contextlib.redirect_stdout(sys.stderr):
                    from llm.qwen3vl import Qwen3VL
                    model = Qwen3VL(size=key[0], res=key[1])
                model_key = key
            reply({'state': 'describing'})
            with contextlib.redirect_stdout(sys.stderr):
                image = None
                if request['image_path'] is not None:
                    import cv2
                    frame = cv2.imread(request['image_path'])
                    if frame is None:
                        raise ValueError('Event image cannot be read')
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                description = model.generate(
                    prompt=request['prompt'], image=image,
                    reset=request['reset'], max_tokens=request['max_tokens'], quiet=True,
                ).strip()
            heartbeat_stop.set()
            reply({'description': description})
        except Exception:
            heartbeat_stop.set()
            reply({'error': traceback.format_exc()})


if __name__ == '__main__':
    serve()
