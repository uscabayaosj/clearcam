"""Exercise real IPC and cache isolation without loading large model weights."""
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from tinygrad.helpers import diskcache_get
from utils.qwen_process import QwenProcess

FAKE_WORKER = '''
import sys, types
from tinygrad.helpers import diskcache_get
class Model:
    def __init__(self, **kwargs):
        diskcache_get('clearcam_cache_probe', 'probe')
    def generate(self, **kwargs):
        diskcache_get('clearcam_cache_probe', 'probe')
        return 'A person is near the door.'
sys.modules['llm.qwen3vl'] = types.SimpleNamespace(Qwen3VL=Model)
from utils.qwen_process import serve
serve()
'''


class QwenProcessTests(unittest.TestCase):
    def test_main_thread_cache_then_background_requests_are_isolated(self):
        diskcache_get('clearcam_cache_probe', 'probe')
        real_popen = subprocess.Popen
        def launch(command, **kwargs):
            self.assertEqual(command[1:3], ['-m', 'utils.qwen_process'])
            return real_popen([sys.executable, '-c', FAKE_WORKER], **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.jpg'
            cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8))
            results, states, errors = [], [], []
            def run():
                with patch('utils.qwen_process.subprocess.Popen', launch):
                    worker = QwenProcess(timeout=10)
                try:
                    for _ in range(2):
                        results.append(worker.generate(image_path=path, prompt='Describe', on_state=states.append))
                except Exception as exc:
                    errors.append(exc)
                finally:
                    worker.close()
                    self.assertIsNotNone(worker.process.poll())
            thread = threading.Thread(target=run)
            thread.start()
            thread.join(timeout=25)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, ['A person is near the door.'] * 2)
            self.assertEqual(states.count('loading_model'), 1)
            self.assertEqual(states.count('describing'), 2)

    def test_timeout_terminates_worker(self):
        real_popen = subprocess.Popen
        def launch(command, **kwargs):
            return real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        with patch('utils.qwen_process.subprocess.Popen', launch):
            worker = QwenProcess(timeout=.1)
        with self.assertRaises(TimeoutError):
            worker.generate(image_path='/unused', prompt='Describe')
        self.assertIsNotNone(worker.process.poll())


if __name__ == '__main__':
    unittest.main()
