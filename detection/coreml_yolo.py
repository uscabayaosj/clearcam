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


def available_sizes(model_dirs):
    """Detector sizes whose Core ML package is actually present."""
    from pathlib import Path
    found = []
    for size, name in MODEL_FILES.items():
        if any(d and (Path(d) / name).exists() for d in model_dirs): found.append(size)
    return found


class CoreMLYolo:
    kind = 'coreml'

    def __init__(self, package_path, confidence=0.25, iou=0.45):
        import coremltools as ct
        from PIL import Image
        self._image = Image
        self.model = ct.models.MLModel(str(package_path))
        self.confidence = confidence
        self.iou = iou

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
        return preds[scores >= self.confidence]
