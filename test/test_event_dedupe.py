import unittest

from utils.event_dedupe import RecentTriggers, iou


def pred(x1, y1, x2, y2, conf, cls):
    return [x1, y1, x2, y2, conf, cls, 0]


LABELS = {0: 'person', 2: 'car'}
label_of = lambda p: LABELS[int(p[5])]


class EventDedupeTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_still_subject_is_not_news_but_a_new_one_beside_it_is(self):
        recent = RecentTriggers(ttl=600)
        parked = pred(1543, 1001, 1896, 1248, 0.9, 2)
        first = recent.novel([parked], label_of, now=0)
        self.assertIs(first, parked)
        recent.remember(first, 'car', now=0)
        # same car, slightly jittered box, seven minutes later: suppressed
        again = pred(1544, 998, 1929, 1247, 0.92, 2)
        self.assertIsNone(recent.novel([again], label_of, now=420))
        # a person appears next to the car: reported, and the person is the trigger
        walker = pred(74, 404, 118, 477, 0.6, 0)
        trigger = recent.novel([again, walker], label_of, now=500)
        self.assertIs(trigger, walker)

    def test_memory_expires_unless_refreshed(self):
        recent = RecentTriggers(ttl=600)
        car = pred(0, 0, 100, 100, 0.9, 2)
        recent.remember(car, 'car', now=0)
        self.assertIsNone(recent.novel([car], label_of, now=500))     # refreshed at 500
        self.assertIsNone(recent.novel([car], label_of, now=1000))    # still within ttl of 500
        self.assertIs(recent.novel([car], label_of, now=1700), car)   # 700 s after last refresh

    def test_same_box_different_label_is_a_different_subject(self):
        recent = RecentTriggers()
        recent.remember(pred(0, 0, 100, 100, 0.9, 2), 'car', now=0)
        person = pred(0, 0, 100, 100, 0.9, 0)
        self.assertIs(recent.novel([person], label_of, now=1), person)


if __name__ == '__main__':
    unittest.main()
