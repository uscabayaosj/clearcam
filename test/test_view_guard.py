import numpy as np
import unittest
from utils import view_guard as vg


def scene(seed, brightness=1.0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(vg.THUMB[1], vg.THUMB[0])).astype(np.float32)
    base = np.clip(base * brightness, 0, 255)
    return base.astype(np.uint8)


class ViewGuardTests(unittest.TestCase):
    def test_turn_away_pauses_and_return_resumes(self):
        g = vg.ViewGuard()
        home, away = scene(1), scene(2)
        t = 0.0
        for _ in range(5): self.assertFalse(g.check(t, home)); t += 1
        moved = [g.check(t + k, away) for k in range(6)]
        self.assertTrue(moved[-1])
        t += 6
        back = [g.check(t + k, home) for k in range(6)]
        self.assertFalse(back[-1])

    def test_gradual_light_change_does_not_pause(self):
        g = vg.ViewGuard()
        rng = np.random.default_rng(3)
        img = scene(1).astype(np.float32)
        for k in range(600):   # slowly fade and add gentle noise, as at dusk
            frame = np.clip(img * (1 - k / 1200) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
            self.assertFalse(g.check(float(k), frame))

    def test_brief_obstruction_is_ignored(self):
        g = vg.ViewGuard()
        home = scene(1)
        for k in range(5): g.check(float(k), home)
        g.check(5.0, scene(9)); g.check(6.0, scene(9))   # 2 s: a bird, a bus
        self.assertFalse(g.check(7.0, home))
