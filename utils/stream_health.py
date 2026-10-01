"""How well a camera's stream is arriving, for the live panel's connection badge.

A camera on a weak Wi-Fi link keeps freezing and reconnecting. The panel can't
fix that, but it can say so instead of looking like the app has stalled.
"""
from collections import deque

FROZEN_AFTER = 3.0         # no new frame for this long = picture is frozen
UNSTABLE_WINDOW = 600.0    # drops are counted over the last ten minutes...
UNSTABLE_DROPS = 3         # ...and this many mark the connection as weak


class StreamHealth:
    def __init__(self):
        self.last_frame = None
        self.drops = deque(maxlen=64)
        self._restarting = False

    def frame(self, now):
        # A picture that froze and came back is a drop, whether the stream
        # recovered by itself (most do) or needed a restart.
        if self._restarting or (self.last_frame is not None and now - self.last_frame > FROZEN_AFTER):
            self.drops.append(now)
        self._restarting = False
        self.last_frame = now

    def restarted(self, now):
        self._restarting = self.last_frame is not None

    def status(self, now):
        """state: 'ok', 'weak' (streaming, but it keeps dropping) or
        'reconnecting' (picture frozen right now)."""
        recent = sum(1 for t in self.drops if now - t <= UNSTABLE_WINDOW)
        age = None if self.last_frame is None else max(0.0, now - self.last_frame)
        if age is not None and age > FROZEN_AFTER:
            state = 'reconnecting'
            recent += 1   # the freeze happening now counts too
        elif recent >= UNSTABLE_DROPS:
            state = 'weak'
        else:
            state = 'ok'
        return {"state": state, "frame_age": None if age is None else round(age, 1), "drops_10m": recent}


class RepeatedPicture:
    """Spot a decoder repeating one picture.

    ffmpeg's rawvideo output pads to a constant frame rate, so when a camera's
    stream freezes it keeps emitting the last picture forever: frames arrive,
    nothing changes, and every "no new frame" watchdog is fooled (tapo360 sat on
    one picture for 2 h 40 min). A camera's on-screen clock changes every
    second, so a repeated picture is never a real new frame.
    """

    def __init__(self, stride=97):
        self.stride = stride
        self.last = None

    def is_new(self, raw_bytes):
        import zlib
        signature = zlib.crc32(bytes(raw_bytes[::self.stride]))
        if signature == self.last:
            return False
        self.last = signature
        return True
