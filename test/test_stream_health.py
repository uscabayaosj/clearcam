import unittest
from utils.stream_health import StreamHealth, FROZEN_AFTER, UNSTABLE_WINDOW


def streaming(h, start, end, step=0.2):
  t = start
  while t <= end:
    h.frame(t)
    t += step


class StreamHealthTest(unittest.TestCase):
  def test_steady_stream_is_ok(self):
    h = StreamHealth()
    streaming(h, 0.0, 60.0)
    s = h.status(60.1)
    self.assertEqual((s["state"], s["drops_10m"]), ("ok", 0))

  def test_no_frame_yet_is_not_reported_frozen(self):
    self.assertEqual(StreamHealth().status(100.0), {"state": "ok", "frame_age": None, "drops_10m": 0})

  def test_frozen_picture_is_reconnecting(self):
    h = StreamHealth()
    h.frame(100.0)
    s = h.status(100.0 + FROZEN_AFTER + 1)
    self.assertEqual((s["state"], s["frame_age"], s["drops_10m"]), ("reconnecting", FROZEN_AFTER + 1, 1))

  def test_freezes_that_recover_by_themselves_count_as_drops(self):
    h = StreamHealth()
    streaming(h, 0.0, 10.0)
    for gap_start in (10.0, 100.0, 200.0):
      streaming(h, gap_start + 8, gap_start + 60)   # an 8 s freeze, then frames again
    self.assertEqual(h.status(260.1)["state"], "weak")
    self.assertEqual(h.status(260.1)["drops_10m"], 3)

  def test_a_restart_counts_as_a_drop(self):
    h = StreamHealth()
    streaming(h, 0.0, 10.0)
    h.restarted(10.5)
    h.frame(11.0)
    self.assertEqual(h.status(11.1)["drops_10m"], 1)

  def test_first_connection_is_not_a_drop(self):
    h = StreamHealth()
    h.restarted(0.0)
    h.frame(1.0)
    self.assertEqual(h.status(1.1)["drops_10m"], 0)

  def test_old_drops_age_out(self):
    h = StreamHealth()
    streaming(h, 0.0, 1.0)
    for t in (10.0, 20.0, 30.0):
      h.frame(t)
    later = 30.0 + UNSTABLE_WINDOW + 5
    streaming(h, 30.0, later)
    self.assertEqual(h.status(later)["state"], "ok")



class RepeatedPictureTest(unittest.TestCase):
  def test_identical_picture_is_not_new(self):
    from utils.stream_health import RepeatedPicture
    r = RepeatedPicture()
    frame = bytes(range(256)) * 1000
    self.assertTrue(r.is_new(frame))
    self.assertFalse(r.is_new(bytes(frame)))

  def test_a_small_change_is_new(self):
    from utils.stream_health import RepeatedPicture
    r = RepeatedPicture(stride=1)
    frame = bytearray(256 * 1000)
    self.assertTrue(r.is_new(bytes(frame)))
    frame[5000] = 1                       # e.g. the camera's clock ticking
    self.assertTrue(r.is_new(bytes(frame)))


if __name__ == "__main__":
  unittest.main()
