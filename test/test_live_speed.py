"""Live view encodes at the viewer's width, once per frame for all viewers;
Keychain lookups are remembered."""
import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import cv2
import numpy as np


def load_live_view_jpeg(draw_calls):
    source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
    wanted = {'live_view_jpeg'}
    body = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    ns = {'cv2': cv2, 'np': np, 'threading': threading, '_live_jpeg_cache': {}, '_live_jpeg_lock': threading.Lock(),
          'draw_live_boxes': lambda frame, boxes, timers=None: (draw_calls.append(boxes.copy()), frame)[1]}
    exec(compile(ast.Module(body=body, type_ignores=[]), '<live>', 'exec'), ns)
    return ns


class LiveViewJpegTests(unittest.TestCase):
    def setUp(self):
        self.draws = []
        self.ns = load_live_view_jpeg(self.draws)
        self.frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        boxes = np.array([[640, 360, 800, 500, 0.9, 2, 1]], dtype=np.float32)
        self.cam = SimpleNamespace(live_boxes={'c': (100.0, boxes)}, count_zones={})

    def decode(self, payload):
        return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

    def test_scaled_to_viewer_width_in_80px_steps(self):
        jpg = self.ns['live_view_jpeg'](self.cam, 'c', self.frame, 1, True, 610, 100.5)
        self.assertEqual(self.decode(jpg).shape[1], 640)
        self.assertAlmostEqual(float(self.draws[0][0, 0]), 320.0)   # boxes scaled with the frame

    def test_never_upscaled_and_default_is_full_size(self):
        self.assertEqual(self.decode(self.ns['live_view_jpeg'](self.cam, 'c', self.frame, 1, False, 4000, 100.5)).shape[1], 1280)
        self.assertEqual(self.decode(self.ns['live_view_jpeg'](self.cam, 'c', self.frame, 2, False, 0, 100.5)).shape[1], 1280)

    def test_one_encode_shared_by_viewers_of_the_same_frame(self):
        with mock.patch.object(cv2, 'imencode', wraps=cv2.imencode) as enc:
            a = self.ns['live_view_jpeg'](self.cam, 'c', self.frame, 7, True, 640, 100.5)
            b = self.ns['live_view_jpeg'](self.cam, 'c', self.frame, 7, True, 640, 100.5)
            self.ns['live_view_jpeg'](self.cam, 'c', self.frame, 8, True, 640, 100.5)
        self.assertIs(a, b)
        self.assertEqual(enc.call_count, 2)


class KeychainCacheTests(unittest.TestCase):
    def test_retrieve_runs_security_once_per_reference(self):
        from utils import keychain
        keychain._resolved.clear()
        ran = SimpleNamespace(stdout='rtsp://u:p@cam/stream\n')
        with mock.patch.object(keychain.subprocess, 'run', return_value=ran) as run:
            ref = keychain.PREFIX + 'cam'
            for _ in range(5):
                self.assertEqual(keychain.retrieve('cam', ref), 'rtsp://u:p@cam/stream')
            self.assertEqual(run.call_count, 1)
            keychain.store('cam', 'rtsp://new')          # a changed address is re-read
            keychain.retrieve('cam', ref)
            self.assertEqual(run.call_count, 3)
        keychain._resolved.clear()
