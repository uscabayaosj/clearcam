"""Notice when a camera's view no longer matches the one its zones were drawn on.

A pan-tilt camera that follows motion or patrols (a Tapo C2x0 "360", say) turns
away and back on its own. Zones are drawn in frame coordinates, so while it
looks elsewhere every parked car seems to leave and arrive again. The guard
compares a small edge map of each frame with a reference and reports the view
as moved while they disagree; counting pauses meanwhile.

Edges rather than pixels, so day/night and passing clouds don't look like a
move; the reference slowly follows the scene while the view is steady.
"""
import numpy as np

THUMB = (64, 36)
MOVED_BELOW = 0.45      # correlation of edge maps under this = a different view
STEADY_ABOVE = 0.65     # ...and over this = the zones' view again
HOLD_SECONDS = 3.0      # a verdict has to last this long before it flips
FOLLOW = 0.02           # reference follows a steady scene this fast per sample
JUMP_BELOW = 0.5        # consecutive samples this different = the camera turned (light changes are gradual)
JUMP_MEMORY = 20.0      # a jump this recent can start a "moved" verdict


def edge_map(gray_small):
    g = gray_small.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    e = gx + gy
    e -= e.mean()
    n = np.linalg.norm(e)
    return e / n if n > 1e-6 else e


def similarity(a, b):
    return float((a * b).sum())


class ViewGuard:
    def __init__(self, reference=None):
        self.reference = reference
        self.moved = False
        self._pending_since = None
        self._previous = None
        self._last_jump = None
        self.moved_since = None
        self.last_score = 1.0

    def reset(self):
        self.__init__()

    def check(self, now, gray_small):
        """gray_small: THUMB-sized grayscale frame. Returns True while the view
        is considered moved away from the reference."""
        e = edge_map(gray_small)
        if self._previous is not None and similarity(e, self._previous) < JUMP_BELOW:
            self._last_jump = now
        self._previous = e
        if self.reference is None:
            self.reference = e
            return False
        score = similarity(e, self.reference)
        self.last_score = score
        if not self.moved:
            # Only an abrupt change starts a "moved" verdict: dusk, rain or
            # streetlights change the scene gradually and the reference follows.
            recent_jump = self._last_jump is not None and now - self._last_jump <= JUMP_MEMORY
            flip = score < MOVED_BELOW and (recent_jump or self._pending_since is not None)
        else:
            flip = score > STEADY_ABOVE
        if flip:
            if self._pending_since is None:
                self._pending_since = now
            elif now - self._pending_since >= HOLD_SECONDS:
                self.moved = not self.moved
                self.moved_since = now if self.moved else None
                self._pending_since = None
        else:
            self._pending_since = None
        if not self.moved and score > STEADY_ABOVE:
            ref = (1 - FOLLOW) * self.reference + FOLLOW * e
            n = np.linalg.norm(ref)
            self.reference = ref / n if n > 1e-6 else ref
        return self.moved
