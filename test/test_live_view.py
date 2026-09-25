"""Unit coverage for draw_live_boxes without loading the heavy clearcam module."""
import ast
from pathlib import Path
import unittest

import numpy as np
import cv2


def load_draw_live_boxes():
    source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
    func = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'draw_live_boxes')
    is_bright = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'is_bright_color')
    namespace = {
        'cv2': cv2,
        'np': np,
        'class_labels': ['person', 'car'],
        'color_dict': {'person': (50, 100, 150), 'car': (100, 200, 50)},
    }
    exec(compile(ast.Module(body=[is_bright, func], type_ignores=[]), '<draw_live_boxes>', 'exec'), namespace)
    return namespace['draw_live_boxes']


class DrawLiveBoxesTests(unittest.TestCase):
    def setUp(self):
        self.draw_live_boxes = load_draw_live_boxes()

    def make_frame(self, h=720, w=1280):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_same_shape_frame(self):
        frame = self.make_frame()
        boxes = np.array([[100, 100, 300, 300, 0.87, 0, 1]], dtype=np.float32)
        out = self.draw_live_boxes(frame, boxes)
        self.assertEqual(out.shape, frame.shape)

    def test_pixels_change_along_box_edge(self):
        frame = self.make_frame()
        boxes = np.array([[100, 100, 300, 300, 0.87, 0, 1]], dtype=np.float32)
        before = frame.copy()
        out = self.draw_live_boxes(frame, boxes)
        # The rectangle is drawn along the box border; the top edge must differ.
        self.assertFalse(np.array_equal(before[100, 100:300], out[100, 100:300]))

    def test_empty_boxes_noop(self):
        frame = self.make_frame()
        before = frame.copy()
        out = self.draw_live_boxes(frame, np.zeros((0, 7), dtype=np.float32))
        self.assertTrue(np.array_equal(before, out))
        self.assertEqual(out.shape, frame.shape)

    def test_out_of_range_class_id_does_not_raise(self):
        frame = self.make_frame()
        before = frame.copy()
        boxes = np.array([[50, 50, 200, 200, 0.5, 99, 7]], dtype=np.float32)
        out = self.draw_live_boxes(frame, boxes)
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(before[50, 50:200], out[50, 50:200]))

    def test_box_touching_frame_edge_does_not_raise(self):
        frame = self.make_frame()
        boxes = np.array([
            [0, 0, 150, 100, 0.9, 1, 2],        # touches top-left corner
            [1100, 600, 1279, 719, 0.6, 0, 3],  # touches bottom-right corner
        ], dtype=np.float32)
        out = self.draw_live_boxes(frame, boxes)
        self.assertEqual(out.shape, frame.shape)


if __name__ == '__main__':
    unittest.main()
