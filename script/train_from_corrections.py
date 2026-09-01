"""Fine-tune the bundled detector on this home's corrections.

Usage:
  <export-venv>/bin/python script/train_from_corrections.py \
      --data "$HOME/Library/Application Support/ClearCam/Data" --size s [--epochs 20] [--teacher yolo11m.pt]

What it does, and why each step exists:
  1. Assembles a YOLO dataset from Data/corrections (owner verdicts) plus
     recent event frames. Owner verdicts are ground truth: 'wrong_label'
     rewrites the triggering box's class, 'not_object' makes the frame a hard
     negative, 'confirm' keeps the boxes as they were.
  2. Optionally asks a stronger teacher (YOLO11m) to relabel every frame and
     keeps only boxes where teacher and student agree, so the student learns
     from information it did not already have. Naive self-training on its own
     outputs would only amplify its own mistakes.
  3. Fine-tunes with the backbone frozen and a small learning rate, so the
     model adapts to these cameras without forgetting the COCO classes.
  4. Exports Core ML as models/yolo11<size>-home.mlpackage; the engine prefers
     a -home package over the stock one of the same size.
Needs the export venv (ultralytics, torch, coremltools). Ultralytics is AGPL;
this is for personal installs unless that is resolved for distribution.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO = [l.strip() for l in (ROOT / 'models' / 'coco.names').read_text().splitlines() if l.strip()] if (ROOT / 'models' / 'coco.names').exists() else None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def to_yolo_line(cls, box, width, height):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def assemble(data_root, out_dir, names, teacher=None, min_corrections=20):
    from PIL import Image
    store = data_root / 'corrections'
    rows = [json.loads(l) for l in (store / 'corrections.jsonl').read_text().splitlines() if l.strip()] if (store / 'corrections.jsonl').exists() else []
    if len(rows) < min_corrections:
        sys.exit(f'Only {len(rows)} corrections recorded; collect at least {min_corrections} (use Wrong on journal events) before training.')
    images_dir, labels_dir = out_dir / 'images', out_dir / 'labels'
    for d in (images_dir, labels_dir): d.mkdir(parents=True, exist_ok=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    counts = dict(confirm=0, wrong_label=0, not_object=0, teacher_boxes=0)
    for row in rows:
        src = store / 'images' / row['image']
        if not src.is_file(): continue
        with Image.open(src) as im: width, height = im.size
        dst = images_dir / src.name
        shutil.copy2(src, dst)
        lines = []
        if row['verdict'] == 'not_object':
            counts['not_object'] += 1          # empty label file = hard negative
        else:
            for det in row.get('detections', []):
                cls = det['cls']
                if row['verdict'] == 'wrong_label' and det.get('trigger'):
                    cls = name_to_idx.get(row['label'], cls)
                lines.append(to_yolo_line(cls, det['box'], width, height))
            counts[row['verdict']] += 1
        if teacher is not None and row['verdict'] != 'not_object':
            # Teacher boxes that agree with an existing box are kept; new
            # teacher-only boxes with high confidence are added.
            preds = teacher.predict(str(src), conf=0.5, verbose=False)[0]
            for tb, tc, tconf in zip(preds.boxes.xyxy.tolist(), preds.boxes.cls.tolist(), preds.boxes.conf.tolist()):
                if any(iou(tb, d['box']) > 0.5 for d in row.get('detections', [])): continue
                if tconf >= 0.6:
                    lines.append(to_yolo_line(int(tc), tb, width, height)); counts['teacher_boxes'] += 1
        (labels_dir / (dst.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
    yaml = out_dir / 'home.yaml'
    yaml.write_text(f"path: {out_dir}\ntrain: images\nval: images\nnames:\n" + ''.join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return yaml, counts, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='ClearCam Data directory')
    parser.add_argument('--size', default='s', choices=['n', 's'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--teacher', default='yolo11m.pt', help="'none' to skip teacher relabeling")
    parser.add_argument('--out', default=None)
    parser.add_argument('--min-corrections', type=int, default=20)
    args = parser.parse_args()
    from ultralytics import YOLO
    data_root = Path(args.data).expanduser()
    work = Path(args.out) if args.out else data_root / 'training' / f'yolo11{args.size}-home'
    if work.exists(): shutil.rmtree(work)
    base = YOLO(f'yolo11{args.size}.pt')
    names = [base.names[i] for i in range(len(base.names))]
    teacher = None if args.teacher == 'none' else YOLO(args.teacher)
    yaml, counts, total = assemble(data_root, work / 'dataset', names, teacher, args.min_corrections)
    print(f'dataset: {total} corrections -> {counts}')
    # Frozen backbone + small LR: adapt to these cameras, keep COCO knowledge.
    base.train(data=str(yaml), epochs=args.epochs, imgsz=640, device='mps', freeze=10, lr0=0.001,
               batch=8, project=str(work), name='run', exist_ok=True, verbose=False, plots=False)
    best = work / 'run' / 'weights' / 'best.pt'
    tuned = YOLO(str(best))
    exported = Path(tuned.export(format='coreml', nms=True, imgsz=640))
    target = ROOT / 'models' / f'yolo11{args.size}-home.mlpackage'
    if target.exists(): shutil.rmtree(target)
    shutil.move(str(exported), str(target))
    # Report how many owner verdicts the tuned model now honours.
    honoured = dict(stock=0, tuned=0, checked=0)
    for row in [json.loads(l) for l in (data_root / 'corrections' / 'corrections.jsonl').read_text().splitlines() if l.strip()]:
        src = data_root / 'corrections' / 'images' / row['image']
        if not src.is_file() or row['verdict'] == 'confirm': continue
        honoured['checked'] += 1
        for key, model in (('stock', base), ('tuned', tuned)):
            preds = model.predict(str(src), conf=0.5, verbose=False)[0]
            labels = {names[int(c)] for c in preds.boxes.cls.tolist()}
            trigger = next((d for d in row.get('detections', []) if d.get('trigger')), None)
            if row['verdict'] == 'not_object' and (trigger is None or trigger['label'] not in labels): honoured[key] += 1
            if row['verdict'] == 'wrong_label' and row['label'] in labels and (trigger is None or trigger['label'] not in labels): honoured[key] += 1
    print(f'owner verdicts honoured: stock {honoured["stock"]}/{honoured["checked"]}, tuned {honoured["tuned"]}/{honoured["checked"]}')
    print(f'exported {target}; the engine will prefer it on next launch. Re-run script/build_and_run.sh to bundle it.')


if __name__ == '__main__':
    main()
