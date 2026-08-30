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

    def generate(self, *, image_path, prompt, reset=True, max_tokens=96, quiet=True, on_state=None):
        request = dict(size=self.size, res=self.res, image_path=str(image_path),
                       prompt=prompt, reset=reset, max_tokens=max_tokens)
        try:
            self.process.stdin.write((json.dumps(request) + '\n').encode())
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            pending = b''
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError('Local Qwen worker exceeded its time limit')
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('Local Qwen worker stopped unexpectedly')
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
    import traceback
    model = None
    model_key = None

    def reply(payload):
        print(json.dumps(payload), flush=True)

    for line in sys.stdin:
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
                import cv2
                frame = cv2.imread(request['image_path'])
                if frame is None:
                    raise ValueError('Event image cannot be read')
                description = model.generate(
                    prompt=request['prompt'], image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    reset=request['reset'], max_tokens=request['max_tokens'], quiet=True,
                ).strip()
            reply({'description': description})
        except Exception:
            reply({'error': traceback.format_exc()})


if __name__ == '__main__':
    serve()
