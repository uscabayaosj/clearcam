"""Core ML object detection on the Apple Neural Engine.

Drop-in for the tinygrad YOLO: takes a BGR frame, returns Nx6 predictions
[x1, y1, x2, y2, score, class_id] in frame coordinates using COCO class ids.
The .mlpackage is exported at build time (see script/export_coreml.py) with an
embedded NMS pipeline, so no decoding or NMS lives here.
"""
import numpy as np

MODEL_FILES = {'t': 'yolo11n.mlpackage', 's': 'yolo11s.mlpackage', 'm': 'yolo11m.mlpackage'}
MODEL_FILE = MODEL_FILES['t']
INPUT_SIZE = 640


def resolve_package(model_dirs, size):
    """The per-home fine-tune wins over the stock package of the same size."""
    from pathlib import Path
    stock = MODEL_FILES.get(size, MODEL_FILE)
    tuned = stock.replace('.mlpackage', '-home.mlpackage')
    for name in (tuned, stock):
        for d in model_dirs:
            if d and (Path(d) / name).exists(): return Path(d) / name
    return None


def _default_coco_names():
    """Best-effort load of models/coco.names relative to this module, so a
    caller that doesn't pass coco_names explicitly still gets one when the
    repo layout is intact."""
    from pathlib import Path
    for candidate in (
        Path(__file__).resolve().parent.parent / 'models' / 'coco.names',
    ):
        if candidate.exists():
            return [l.strip() for l in candidate.read_text().splitlines() if l.strip()]
    return None


def build_canonical(model_names, coco_names, extras=('stroller', 'child', 'scooter')):
    """Map a model's own class order onto ClearCam's stable canonical vocabulary.

    canonical = coco_names (in COCO order) + extras (in the given fixed
    order) + any other names the model has that don't normalise onto either
    of those, appended in sorted order. This keeps saved alert rules (which
    store class IDS) meaning the same thing regardless of the order a
    Roboflow-trained model happens to list its classes in.

    Returns (canonical_names, index_map) where index_map maps each index of
    model_names to its index in canonical_names. A model name that doesn't
    normalise onto anything in canonical is simply omitted from index_map
    (its predictions get dropped by the caller).
    """
    from utils.roboflow_sync import normalise_class

    canonical = list(coco_names)
    for name in extras:
        name = (name or '').strip()
        if name and name.lower() not in {c.lower() for c in canonical}:
            canonical.append(name)

    # Names the model has that don't normalise onto coco_names+extras yet -
    # collected first, then appended in sorted order (after normalising them
    # against the canonical list built so far).
    leftover = set()
    for name in model_names:
        norm = normalise_class(name, canonical)
        if norm.lower() not in {c.lower() for c in canonical}:
            leftover.add(norm)
    canonical.extend(sorted(leftover))

    lower_pos = {c.lower(): i for i, c in enumerate(canonical)}
    index_map = {}
    for i, name in enumerate(model_names):
        norm = normalise_class(name, canonical)
        pos = lower_pos.get(norm.lower())
        if pos is not None:
            index_map[i] = pos
    return canonical, index_map


def remap_class_ids(preds, lookup):
    """Vectorised remap of column-5 class ids in an Nx6 preds array through
    `lookup` (a 1D array where lookup[model_idx] = canonical_idx, or -1 for
    'drop this prediction'). Rows that map to -1 are dropped."""
    if preds.shape[0] == 0:
        return preds
    class_ids = preds[:, 5].astype(np.int64)
    in_range = (class_ids >= 0) & (class_ids < len(lookup))
    mapped = np.full(class_ids.shape, -1, dtype=np.int64)
    mapped[in_range] = lookup[class_ids[in_range]]
    keep = mapped >= 0
    out = preds[keep].copy()
    out[:, 5] = mapped[keep].astype(np.float32)
    return out


def available_sizes(model_dirs):
    """Detector sizes whose Core ML package is actually present."""
    from pathlib import Path
    found = []
    for size, name in MODEL_FILES.items():
        if any(d and (Path(d) / name).exists() for d in model_dirs): found.append(size)
    return found


class CoreMLYolo:
    kind = 'coreml'

    def __init__(self, package_path, confidence=0.25, iou=0.45, coco_names=None):
        import coremltools as ct
        from PIL import Image
        self._image = Image
        self.model = ct.models.MLModel(str(package_path))
        self.confidence = confidence
        self.iou = iou
        self.names = None
        try:
            import ast
            raw = self.model.user_defined_metadata.get('names')
            if raw:
                parsed = ast.literal_eval(raw)
                self.names = [parsed[i] for i in sorted(parsed, key=int)]
        except Exception:
            self.names = None

        self.canonical_names = None
        self._class_lookup = None  # None means identity (fast path)
        if self.names:
            coco = coco_names if coco_names is not None else _default_coco_names()
            if coco:
                canonical, index_map = build_canonical(self.names, coco)
                self.canonical_names = canonical
                if self.names == canonical[:len(self.names)] and all(
                    index_map.get(i) == i for i in range(len(self.names))
                ):
                    # Stock COCO model (or already-canonical order): identity map,
                    # skip the per-frame remap entirely.
                    self._class_lookup = None
                else:
                    lookup = np.full(len(self.names), -1, dtype=np.int64)
                    for src, dst in index_map.items():
                        lookup[src] = dst
                    self._class_lookup = lookup

    def __call__(self, frame_bgr):
        frame_bgr = np.asarray(frame_bgr)
        height, width = frame_bgr.shape[:2]
        scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
        new_w, new_h = round(width * scale), round(height * scale)
        pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        import cv2
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized[:, :, ::-1]
        output = self.model.predict({
            'image': self._image.fromarray(canvas),
            'confidenceThreshold': self.confidence,
            'iouThreshold': self.iou,
        })
        confidence = np.asarray(output['confidence'])
        coordinates = np.asarray(output['coordinates'])
        if confidence.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        class_ids = confidence.argmax(axis=1)
        scores = confidence.max(axis=1)
        # Normalized center xywh on the letterboxed square -> frame corner coords.
        cx, cy, w, h = (coordinates * INPUT_SIZE).T
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        preds = np.stack([
            x1.clip(0, width), y1.clip(0, height),
            x2.clip(0, width), y2.clip(0, height),
            scores, class_ids.astype(np.float32),
        ], axis=1).astype(np.float32)
        preds = preds[scores >= self.confidence]
        if self._class_lookup is not None:
            preds = remap_class_ids(preds, self._class_lookup)
        return preds
