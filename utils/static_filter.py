"""Drop detections that never move on a fixed camera.

At night even a large detector sees 'people' in posts, bollards and shadows,
and it sees them at the same spot in frame after frame. Real people, bikes
and dogs move. So a movable-class box that recurs at the same place in more
than a small fraction of one camera's frames is treated as scenery and
dropped. Parked cars and trucks legitimately stay put and are left alone.
"""
import re

MOVABLE = frozenset({'person', 'bicycle', 'dog', 'stroller', 'child', 'scooter'})


def camera_of(filename):
    """'own-<camera>-YYYYMMDD-HHMMSS.jpg' -> '<camera>' (whole name otherwise)."""
    m = re.match(r'own-(.+)-\d{8}-\d{6}\.\w+$', filename)
    return m.group(1) if m else filename


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def static_mask(frames, max_fraction=0.1, min_iou=0.5, movable=MOVABLE):
    """frames: list of (camera, [(class_name, (x1, y1, x2, y2)), ...]).

    Returns, per frame, a list of booleans: True where the box should be kept.
    Coordinates may be pixels or normalised, as long as one camera is consistent.
    """
    by_camera = {}
    for i, (camera, _boxes) in enumerate(frames):
        by_camera.setdefault(camera, []).append(i)
    keep = [[True] * len(boxes) for _camera, boxes in frames]
    for indices in by_camera.values():
        if len(indices) < 5:
            continue   # too few frames to tell scenery from a visitor
        limit = max_fraction * len(indices)
        for i in indices:
            for j, (name, box) in enumerate(frames[i][1]):
                if name not in movable:
                    continue
                recurs = sum(
                    1 for k in indices if k != i and any(
                        other_name == name and iou(box, other) >= min_iou for other_name, other in frames[k][1]))
                if recurs > limit:
                    keep[i][j] = False
    return keep


def parse_ignore(specs):
    """--ignore 'camera=x1,y1,x2,y2[:class,class]' (normalised coords) ->
    {camera: [((x1, y1, x2, y2), frozenset(classes) or None)]}."""
    areas = {}
    for spec in specs or []:
        camera, sep, rest = spec.partition('=')
        box_part, _, class_part = rest.partition(':')
        try:
            box = tuple(float(v) for v in box_part.split(','))
        except ValueError:
            box = ()
        if not sep or not camera.strip() or len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f'invalid --ignore {spec!r}: expected camera=x1,y1,x2,y2[:class,...] with 0-1 coords')
        classes = frozenset(c.strip() for c in class_part.split(',') if c.strip()) or None
        areas.setdefault(camera.strip(), []).append((box, classes))
    return areas


def ignored(camera, name, box, areas):
    """True when the box centre lies in one of this camera's ignore areas covering its class."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any((classes is None or name in classes) and a[0] <= cx <= a[2] and a[1] <= cy <= a[3]
               for a, classes in areas.get(camera, ()))
