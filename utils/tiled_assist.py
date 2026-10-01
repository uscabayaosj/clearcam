"""Run the vehicle-assist detector on two halves of a wide frame, for pushchairs.

The assist model sees a 1280x720 frame shrunk to 640 wide, where a pushchair
is ~30 px - too small for a Nano model to find (0 of 45 on the owner's
cameras). Run on two overlapping square halves it sees them about twice as
big (13 of 45 at the app's 0.4 threshold, 3 false ones in 250 empty frames).

Only the tile classes (stroller) come from the halves; cars and trucks keep
coming from the whole-frame pass, which already finds them well, so a car
cut in two by a tile edge can't be counted twice.
"""
import numpy as np

STROLLER_ID = 80                 # canonical id (COCO order + extras), see assist_merge
TILE_CLASSES = frozenset({STROLLER_ID})
MIN_ASPECT = 1.2                 # narrower frames already give objects enough pixels
NMS_IOU = 0.5


def tile_offsets(width, height):
    """x offsets of square height x height tiles covering a wide frame: one at
    each edge (they overlap in the middle). [] when tiling wouldn't help."""
    if height <= 0 or width < height * MIN_ASPECT:
        return []
    return [0, width - height]


def _iou(box, boxes):
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def nms(preds, iou=NMS_IOU):
    """Per-class greedy NMS over Nx6 [x1, y1, x2, y2, score, class] rows."""
    if len(preds) < 2:
        return preds
    keep = []
    for cls in np.unique(preds[:, 5]):
        rows = np.where(preds[:, 5] == cls)[0]
        rows = rows[np.argsort(-preds[rows, 4])]
        while len(rows):
            best, rows = rows[0], rows[1:]
            keep.append(best)
            if len(rows):
                rows = rows[_iou(preds[best, :4], preds[rows, :4]) <= iou]
    return preds[sorted(keep)]


def _only(preds, ids):
    return preds[np.isin(preds[:, 5], ids)]


def assist_predict(detector, frame, tile_classes=TILE_CLASSES, pushchair_detector=None):
    """Whole-frame assist predictions, with tile_classes also searched for in
    two half-frame squares. Returns Nx6 rows in whole-frame pixels.

    pushchair_detector, when given, owns tile_classes entirely (whole frame and
    halves) and `detector` keeps only its other classes: the half-frame-trained
    pushchair model finds 35/35 held-back pushchairs but only 63% of cars,
    while the vehicle model keeps cars at 93%."""
    rows = lambda d, img: np.asarray(d(img), dtype=np.float32).reshape(-1, 6)
    full = rows(detector, frame)
    ids = np.array(sorted(tile_classes), dtype=np.float32)
    source = detector
    if pushchair_detector is not None:
        # The two halves cover the whole frame, so the pushchair model needs no
        # whole-frame pass of its own (one fewer inference per frame).
        full = full[~np.isin(full[:, 5], ids)]
        source = pushchair_detector
    height, width = frame.shape[:2]
    offsets = tile_offsets(width, height)
    if not offsets or not tile_classes:
        if pushchair_detector is not None:   # too narrow to split: one whole-frame pass
            full = np.concatenate([full, _only(rows(pushchair_detector, frame), ids)], axis=0)
        return full
    parts = [full[np.isin(full[:, 5], ids)]]
    for x0 in offsets:
        tile = rows(source, frame[:, x0:x0 + height])
        tile = tile[np.isin(tile[:, 5], ids)].copy()
        tile[:, [0, 2]] += x0
        parts.append(tile)
    found = nms(np.concatenate(parts, axis=0))
    return np.concatenate([full[~np.isin(full[:, 5], ids)], found], axis=0)
