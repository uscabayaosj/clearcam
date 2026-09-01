"""Owner corrections: the only signal a detector cannot get from itself.

Each event records the detections that fired it; the owner can then mark an
event as a wrong label, not an object at all, or confirmed. Corrections are
copied out of the retention-managed camera tree into Data/corrections so they
outlive the recordings, and script/train_from_corrections.py turns them into
a per-home fine-tune of the bundled Core ML detector.
"""
import json
import shutil
import time
from pathlib import Path

VERDICTS = ('confirm', 'wrong_label', 'not_object')


def detections_sidecar(event_path):
    return Path(str(event_path)).with_suffix('.detections.json')


def write_detections(event_path, preds, class_labels, width, height, trigger_index=None):
    """Persist boxes (frame coords) for the event so corrections carry geometry."""
    rows = []
    for index, pred in enumerate(preds):
        x1, y1, x2, y2, score, cls = [float(v) for v in pred[:6]]
        label = class_labels[int(cls)] if int(cls) < len(class_labels) else str(int(cls))
        rows.append(dict(box=[x1, y1, x2, y2], score=score, cls=int(cls), label=label,
                         trigger=(index == trigger_index)))
    payload = dict(width=int(width), height=int(height), detections=rows)
    target = detections_sidecar(event_path)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload))
    temporary.replace(target)
    return target


def read_detections(event_path):
    try:
        return json.loads(detections_sidecar(event_path).read_text())
    except (OSError, ValueError):
        return None


def read_correction(event_path):
    try:
        return json.loads(Path(str(event_path)).with_suffix('.correction.json').read_text())
    except (OSError, ValueError):
        return None


def record_correction(data_root, event_path, verdict, label=None):
    """Store the owner's verdict beside the event and in the durable corrections set."""
    if verdict not in VERDICTS: raise ValueError('Unknown verdict')
    if verdict == 'wrong_label' and not label: raise ValueError('A corrected label is required')
    event_path = Path(event_path)
    if not event_path.is_file(): raise ValueError('Event image not found')
    detections = read_detections(event_path) or dict(width=None, height=None, detections=[])
    store = Path(data_root) / 'corrections'
    images = store / 'images'
    images.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    kept_image = images / f'{stamp}_{event_path.name}'
    shutil.copy2(event_path, kept_image)
    crop = event_path.with_suffix('.trigger.jpg')
    kept_crop = None
    if crop.is_file():
        kept_crop = images / f'{stamp}_{crop.name}'
        shutil.copy2(crop, kept_crop)
    entry = dict(time=time.time(), verdict=verdict, label=label, image=kept_image.name,
                 crop=kept_crop.name if kept_crop else None, camera=event_path.parts[-4] if len(event_path.parts) > 4 else None,
                 width=detections.get('width'), height=detections.get('height'),
                 detections=detections.get('detections', []))
    with (store / 'corrections.jsonl').open('a') as stream:
        stream.write(json.dumps(entry) + '\n')
    sidecar = event_path.with_suffix('.correction.json')
    temporary = sidecar.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(verdict=verdict, label=label, time=entry['time'])))
    temporary.replace(sidecar)
    return entry


def load_corrections(data_root):
    path = Path(data_root) / 'corrections' / 'corrections.jsonl'
    if not path.is_file(): return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows
