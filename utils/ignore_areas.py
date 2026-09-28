"""Per-camera ignore areas for the live detection pipeline.

A rectangle (normalised 0-1 coords of the frame) where detections are
discarded before the tracker ever sees them -- as if the detector never
looked there. Built for cases like camera 'tapoc113', where two rubbish
bins in the lower-right corner get called 'car' in roughly 40% of frames:
one ignore area over the bins, restricted to the 'car' class, and every
downstream consumer (live overlay, per-camera counts/scene descriptions,
car alerts) simply never sees those detections.

Unrelated to utils/static_filter.py, which does a similar-looking job for
the *offline* own-camera training pipeline (script/collect_frames.py,
script/compare_models.py, script/roboflow_dataset.py) and is not touched
here.
"""
import numpy as np

MAX_AREAS = 20


def parse_areas(raw, class_names=None):
    """Validate and normalise a list of ignore-area specs.

    raw: a list of {"box": [x1, y1, x2, y2], "classes": [<name>, ...]}.
    "classes" is optional; omitted or empty means every class.
    class_names: the known class-name vocabulary to validate "classes"
    against (e.g. the engine's `class_labels`), or None to skip that check
    (used by tests that don't care about class validation).

    Returns a list of {"box": [x1, y1, x2, y2], "classes": set(<lower-case
    name>) | None}. Raises ValueError with a human-readable message on any
    invalid input; raises nothing for `raw` being None or [] (both -> []).
    """
    if raw is None or raw == []:
        return []
    if not isinstance(raw, list):
        raise ValueError("ignore_areas must be a list")
    if len(raw) > MAX_AREAS:
        raise ValueError(f"at most {MAX_AREAS} ignore areas are allowed")

    known = {c.lower() for c in class_names} if class_names else None
    areas = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each ignore area must be an object")
        box = entry.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("each ignore area needs a box of [x1, y1, x2, y2]")
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            raise ValueError("ignore area box values must be numbers")
        if not (0 <= x1 < x2 <= 1):
            raise ValueError("ignore area box needs 0 <= x1 < x2 <= 1")
        if not (0 <= y1 < y2 <= 1):
            raise ValueError("ignore area box needs 0 <= y1 < y2 <= 1")

        classes = entry.get("classes") or []
        if not isinstance(classes, list):
            raise ValueError("ignore area classes must be a list of class names")
        normalised = set()
        for c in classes:
            if not isinstance(c, str) or not c.strip():
                raise ValueError("ignore area class names must be non-empty strings")
            name = c.strip().lower()
            if known is not None and name not in known:
                raise ValueError(f"unknown class name: {c}")
            normalised.add(name)

        areas.append({"box": [x1, y1, x2, y2], "classes": normalised or None})
    return areas


def filter_preds(preds, areas, width, height, class_names=None):
    """Drop rows of `preds` whose box centre lies inside an ignore area
    that covers that row's class.

    preds: Nx6 array-like [x1, y1, x2, y2, score, class_id] in pixel
    coordinates of the frame that produced them (the shape the detector
    hands to the tracker).
    areas: the normalised list `parse_areas` returns.
    width, height: pixel dimensions of that frame, to un-normalise `areas`.
    class_names: names indexed by class_id, to resolve class-restricted
    areas; if None, class-restricted areas never match (nothing is known
    to compare against) and only unrestricted (all-class) areas apply.

    A no-op (returns preds unchanged) when areas is empty or preds holds
    no rows.
    """
    if preds is None or not areas:
        return preds
    preds = np.asarray(preds)
    if preds.ndim != 2 or preds.shape[0] == 0:
        return preds

    cx = (preds[:, 0] + preds[:, 2]) / 2.0
    cy = (preds[:, 1] + preds[:, 3]) / 2.0
    nx = cx / float(width) if width else cx
    ny = cy / float(height) if height else cy

    drop = np.zeros(preds.shape[0], dtype=bool)
    for area in areas:
        x1, y1, x2, y2 = area["box"]
        inside = (nx >= x1) & (nx <= x2) & (ny >= y1) & (ny <= y2)
        if not inside.any():
            continue
        if area["classes"] is None:
            drop |= inside
            continue
        for i in np.nonzero(inside)[0]:
            cls_id = int(preds[i, 5])
            name = class_names[cls_id].lower() if class_names and 0 <= cls_id < len(class_names) else None
            if name is not None and name in area["classes"]:
                drop[i] = True
    return preds[~drop]
