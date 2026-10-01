import unittest
import numpy as np
from utils import tiled_assist as ta

CAR, STROLLER = 2, ta.STROLLER_ID


class FakeDetector:
  """Returns canned rows per input width: the whole frame (1280) or a tile (720)."""
  def __init__(self, full, tile_left, tile_right):
    self.full, self.left, self.right = full, tile_left, tile_right
    self.calls = []

  def __call__(self, frame):
    self.calls.append(frame.shape[:2])
    if frame.shape[1] == 1280:
      return np.array(self.full, np.float32).reshape(-1, 6)
    # the two tiles are told apart by a marker pixel painted by the test
    return np.array(self.left if frame[0, 0, 0] == 1 else self.right, np.float32).reshape(-1, 6)


def frame():
  f = np.zeros((720, 1280, 3), np.uint8)
  f[0, 0, 0] = 1          # left tile starts at x=0
  return f


class TileOffsetsTest(unittest.TestCase):
  def test_wide_frame_gets_two_overlapping_squares(self):
    self.assertEqual(ta.tile_offsets(1280, 720), [0, 560])

  def test_near_square_frame_is_not_tiled(self):
    self.assertEqual(ta.tile_offsets(800, 720), [])


class AssistPredictTest(unittest.TestCase):
  def test_pushchair_found_only_in_a_tile_is_added_in_frame_coordinates(self):
    det = FakeDetector(full=[[100, 100, 300, 200, 0.75, CAR]], tile_left=[], tile_right=[[100, 300, 140, 360, 0.5, STROLLER]])
    out = ta.assist_predict(det, frame())
    self.assertEqual(len(det.calls), 3)
    strollers = out[out[:, 5] == STROLLER]
    self.assertEqual(strollers.tolist(), [[660, 300, 700, 360, 0.5, STROLLER]])
    self.assertEqual(out[out[:, 5] == CAR].tolist(), [[100, 100, 300, 200, 0.75, CAR]])

  def test_pushchair_in_the_overlap_is_kept_once(self):
    same_spot = [[40, 300, 100, 360, 0.625, STROLLER]]        # x 600-660 in the right tile
    det = FakeDetector(full=[], tile_left=[[600, 300, 660, 360, 0.5, STROLLER]], tile_right=same_spot)
    out = ta.assist_predict(det, frame())
    self.assertEqual(out.tolist(), [[600, 300, 660, 360, 0.625, STROLLER]])

  def test_cars_from_tiles_are_ignored(self):
    det = FakeDetector(full=[], tile_left=[[0, 0, 50, 50, 0.75, CAR]], tile_right=[[0, 0, 50, 50, 0.75, CAR]])
    self.assertEqual(len(ta.assist_predict(det, frame())), 0)

  def test_narrow_frame_is_a_single_pass(self):
    calls = []
    def det(f):
      calls.append(f.shape)
      return np.array([[1, 1, 5, 5, 0.75, STROLLER]], np.float32)
    out = ta.assist_predict(det, np.zeros((720, 800, 3), np.uint8))
    self.assertEqual((len(calls), out.tolist()), (1, [[1, 1, 5, 5, 0.75, STROLLER]]))


class SplitDetectorsTest(unittest.TestCase):
  def test_vehicles_from_one_model_pushchairs_from_the_other(self):
    vehicles = lambda f: np.array([[100, 100, 300, 200, 0.75, CAR], [50, 50, 60, 60, 0.75, STROLLER]], np.float32)
    calls = []
    def prams(f):
      calls.append(f.shape[1])
      return np.array([[10, 300, 40, 360, 0.5, STROLLER], [0, 0, 50, 50, 0.75, CAR]], np.float32) if f.shape[1] == 720 else np.zeros((0, 6), np.float32)
    out = ta.assist_predict(vehicles, frame(), pushchair_detector=prams)
    self.assertEqual(out[out[:, 5] == CAR].tolist(), [[100, 100, 300, 200, 0.75, CAR]])   # only the vehicle model's car
    strollers = sorted(out[out[:, 5] == STROLLER].tolist())
    self.assertEqual(strollers, [[10, 300, 40, 360, 0.5, STROLLER], [570, 300, 600, 360, 0.5, STROLLER]])
    self.assertEqual(sorted(calls), [720, 720])   # the two halves only: they cover the whole frame


class NarrowFrameSplitDetectorsTest(unittest.TestCase):
  def test_narrow_frame_still_searched_by_the_pushchair_model(self):
    vehicles = lambda f: np.array([[1, 1, 5, 5, 0.75, STROLLER]], np.float32)
    prams = lambda f: np.array([[2, 2, 6, 6, 0.5, STROLLER]], np.float32)
    out = ta.assist_predict(vehicles, np.zeros((720, 800, 3), np.uint8), pushchair_detector=prams)
    self.assertEqual(out.tolist(), [[2, 2, 6, 6, 0.5, STROLLER]])


class NmsTest(unittest.TestCase):
  def test_different_classes_never_suppress_each_other(self):
    p = np.array([[0, 0, 10, 10, 0.75, CAR], [0, 0, 10, 10, 0.25, STROLLER]], np.float32)
    self.assertEqual(len(ta.nms(p)), 2)


if __name__ == "__main__":
  unittest.main()
