"""Manual real-model smoke test using synthetic imagery, never camera footage.

Run from repository root: PYTHONPATH=. .venv/bin/python test/local_qwen_smoke.py
"""
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
from utils.local_descriptions import LocalDescriptions, read_description

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='clearcam-qwen-') as directory:
        path = Path(directory) / 'synthetic.jpg'
        canvas = np.full((448, 448, 3), 255, dtype=np.uint8)
        cv2.rectangle(canvas, (110, 110), (338, 338), (0, 0, 220), -1)
        cv2.imwrite(str(path), canvas)
        worker = LocalDescriptions()
        worker.configure(True, 2)
        assert worker.submit(path, 'Synthetic test', notify=False)
        deadline = time.monotonic() + 1200
        last_state = None
        while time.monotonic() < deadline:
            status = worker.status()
            if status['state'] != last_state:
                print(status, flush=True)
                last_state = status['state']
            if status['state'] == 'error':
                raise RuntimeError(status['error']) from worker.failure
            result = read_description(path)
            if result:
                print('Synthetic image description:', result, flush=True)
                break
            time.sleep(2)
        else:
            raise TimeoutError('Local model smoke test exceeded 20 minutes')
