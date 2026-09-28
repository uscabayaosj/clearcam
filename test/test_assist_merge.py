import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.assist_merge import merge_assist, DEFAULT_TAKE_IDS, DEFAULT_MIN_CONFIDENCE

# canonical ids used by the merge (COCO order + fixed extras): person=0,
# car=2, dog=16, truck=7, stroller=80.
PERSON, CAR, DOG, TRUCK, STROLLER = 0, 2, 16, 7, 80


def row(x1, y1, x2, y2, score, cls):
    return [x1, y1, x2, y2, score, cls]


class MergeAssistTests(unittest.TestCase):
    def test_primary_cars_dropped_assist_cars_added(self):
        primary = np.array([
            row(0, 0, 10, 10, 0.9, PERSON),
            row(0, 0, 10, 10, 0.6, CAR),  # should be dropped in favour of assist
        ], dtype=np.float32)
        assist = np.array([
            row(1, 1, 11, 11, 0.8, CAR),
        ], dtype=np.float32)
        out = merge_assist(primary, assist)
        classes = sorted(out[:, 5].astype(int).tolist())
        self.assertEqual(classes, [PERSON, CAR])
        # the surviving car row is assist's, not primary's
        car_row = out[out[:, 5].astype(int) == CAR][0]
        self.assertEqual(list(car_row[:4]), [1, 1, 11, 11])

    def test_assist_trucks_and_strollers_added(self):
        primary = np.zeros((0, 6), dtype=np.float32)
        assist = np.array([
            row(0, 0, 5, 5, 0.9, TRUCK),
            row(0, 0, 5, 5, 0.9, STROLLER),
        ], dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(sorted(out[:, 5].astype(int).tolist()), [TRUCK, STROLLER])

    def test_assist_persons_ignored(self):
        primary = np.array([row(0, 0, 5, 5, 0.9, PERSON)], dtype=np.float32)
        assist = np.array([row(0, 0, 5, 5, 0.95, PERSON)], dtype=np.float32)
        out = merge_assist(primary, assist)
        # assist's person prediction is not in take_ids, so it's dropped and
        # primary's person prediction (never removed) is the only survivor.
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(int(out[0, 5]), PERSON)
        self.assertEqual(list(out[0, :4]), [0, 0, 5, 5])

    def test_assist_dog_and_bicycle_ignored(self):
        primary = np.array([row(0, 0, 5, 5, 0.9, DOG)], dtype=np.float32)
        assist = np.array([row(9, 9, 20, 20, 0.99, DOG)], dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(list(out[0, :4]), [0, 0, 5, 5])

    def test_low_confidence_assist_row_dropped(self):
        primary = np.zeros((0, 6), dtype=np.float32)
        assist = np.array([
            row(0, 0, 5, 5, 0.39, CAR),  # below default 0.4 threshold
            row(0, 0, 5, 5, 0.41, CAR),
        ], dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(out.shape[0], 1)
        self.assertAlmostEqual(float(out[0, 4]), 0.41, places=5)

    def test_min_confidence_is_overridable(self):
        primary = np.zeros((0, 6), dtype=np.float32)
        assist = np.array([row(0, 0, 5, 5, 0.2, CAR)], dtype=np.float32)
        out_default = merge_assist(primary, assist)
        self.assertEqual(out_default.shape[0], 0)
        out_lower = merge_assist(primary, assist, min_confidence=0.1)
        self.assertEqual(out_lower.shape[0], 1)

    def test_take_ids_overridable(self):
        primary = np.array([row(0, 0, 5, 5, 0.9, DOG)], dtype=np.float32)
        assist = np.array([row(0, 0, 5, 5, 0.9, DOG)], dtype=np.float32)
        out = merge_assist(primary, assist, take_ids={DOG})
        # with DOG in take_ids, primary's dog is dropped and assist's kept.
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(int(out[0, 5]), DOG)

    def test_both_empty(self):
        primary = np.zeros((0, 6), dtype=np.float32)
        assist = np.zeros((0, 6), dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(out.shape, (0, 6))
        self.assertEqual(out.dtype, np.float32)

    def test_primary_empty_assist_has_rows(self):
        primary = np.zeros((0, 6), dtype=np.float32)
        assist = np.array([row(0, 0, 5, 5, 0.9, CAR)], dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(out.shape, (1, 6))

    def test_assist_empty_primary_has_rows(self):
        primary = np.array([row(0, 0, 5, 5, 0.9, PERSON)], dtype=np.float32)
        assist = np.zeros((0, 6), dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(out.shape, (1, 6))
        self.assertEqual(int(out[0, 5]), PERSON)

    def test_output_dtype_and_shape_match_primary(self):
        primary = np.array([row(0, 0, 5, 5, 0.9, PERSON)], dtype=np.float32)
        assist = np.array([row(0, 0, 5, 5, 0.9, CAR)], dtype=np.float32)
        out = merge_assist(primary, assist)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[1], 6)

    def test_default_take_ids_are_car_truck_stroller(self):
        self.assertEqual(DEFAULT_TAKE_IDS, frozenset({2, 7, 80}))

    def test_default_min_confidence(self):
        self.assertEqual(DEFAULT_MIN_CONFIDENCE, 0.4)


if __name__ == '__main__':
    unittest.main()
