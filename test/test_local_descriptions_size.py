import unittest
from utils.local_descriptions import usable_size


class UsableSizeTest(unittest.TestCase):
  def test_available_size_is_kept(self):
    self.assertEqual(usable_size(8, [2, 8]), (8, None))

  def test_missing_size_falls_back_to_smallest_with_a_note(self):
    size, note = usable_size(8, [2])
    self.assertEqual(size, 2)
    self.assertIn('8B', note)

  def test_unknown_availability_keeps_the_request(self):
    self.assertEqual(usable_size(4, []), (4, None))


if __name__ == "__main__":
  unittest.main()
