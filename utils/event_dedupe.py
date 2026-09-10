"""Stop a subject that has not moved from filing the same event again.

The tracker drops and re-acquires a still subject (a person in bed, a parked
car), each re-acquired track counts as a new sighting, and the journal fills
with the same scene. Events are deduplicated on what actually triggered
them: a detection whose box still overlaps a recent trigger of the same
label is "still there", not news. A genuinely new object next to it is still
reported, and becomes the trigger the description leads with.
"""
import time

DEFAULT_TTL = 600        # seconds a remembered trigger stays "recent"
DEFAULT_OVERLAP = 0.6    # IoU at which two boxes are the same subject


def iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0: return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class RecentTriggers:
    """Per-camera memory of what recently fired an event."""

    def __init__(self, ttl=DEFAULT_TTL, overlap=DEFAULT_OVERLAP):
        self.ttl, self.overlap = ttl, overlap
        self.entries = []   # dicts: box, label, seen

    def _prune(self, now):
        self.entries = [e for e in self.entries if now - e['seen'] <= self.ttl]

    def _match(self, pred, label):
        for entry in self.entries:
            if entry['label'] == label and iou(entry['box'], pred) >= self.overlap:
                return entry
        return None

    def novel(self, preds, label_of, now=None):
        """The most confident detection that is not a recent trigger, or None.

        When every detection is something already reported, the matching
        memories are refreshed (the subject is still there) and None is
        returned so the caller can skip the event.
        """
        now = time.time() if now is None else now
        self._prune(now)
        ranked = sorted(preds, key=lambda p: -float(p[4]))
        duplicates = []
        for pred in ranked:
            match = self._match(pred, label_of(pred))
            if match is None: return pred
            duplicates.append(match)
        for match in duplicates: match['seen'] = now
        return None

    def remember(self, pred, label, now=None):
        now = time.time() if now is None else now
        match = self._match(pred, label)
        if match is None:
            self.entries.append(dict(box=[float(v) for v in pred[:4]], label=label, seen=now))
        else:
            match['box'], match['seen'] = [float(v) for v in pred[:4]], now
