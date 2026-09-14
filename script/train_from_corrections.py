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
sys.path.insert(0, str(ROOT))
from utils import roboflow_sync
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


def assemble(data_root, out_dir, names, teacher=None, min_corrections=20, skip_gate=False):
    from PIL import Image
    store = data_root / 'corrections'
    rows = [json.loads(l) for l in (store / 'corrections.jsonl').read_text().splitlines() if l.strip()] if (store / 'corrections.jsonl').exists() else []
    disagreements = sum(1 for r in rows if r['verdict'] != 'confirm')
    # "Looks right" alone is the model grading its own homework: only verdicts
    # that disagree with it carry information the fine-tune can learn from.
    # When a Roboflow dataset is supplied there is external data to train on,
    # so the local-corrections gate is informational only, not a hard stop.
    if not skip_gate and (len(rows) < min_corrections or disagreements < max(1, min_corrections // 2)):
        sys.exit(f'{len(rows)} corrections recorded, {disagreements} of them disagreements (Not a … / It was actually …). '
                 f'Training needs at least {min_corrections} in total with {max(1, min_corrections // 2)} disagreements; '
                 'confirmations alone would only reinforce what the detector already believes.')
    print(f'local corrections: {len(rows)} recorded, {disagreements} disagreements')
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
    parser.add_argument('--roboflow-version', type=int, default=None,
                         help='Pull this Roboflow dataset version and merge it with the local corrections.')
    args = parser.parse_args()
    from ultralytics import YOLO
    data_root = Path(args.data).expanduser()
    work = Path(args.out) if args.out else data_root / 'training' / f'yolo11{args.size}-home'
    if work.exists(): shutil.rmtree(work)
    base = YOLO(f'yolo11{args.size}.pt')
    names = [base.names[i] for i in range(len(base.names))]
    teacher = None if args.teacher == 'none' else YOLO(args.teacher)

    roboflow_train_dir = roboflow_val_dir = None
    if args.roboflow_version is not None:
        cfg = roboflow_sync.load_config(data_root)
        missing = [k for k in ('api_key', 'workspace', 'project') if not cfg.get(k)]
        if missing:
            sys.exit(f'--roboflow-version needs Roboflow config: missing {", ".join(missing)} '
                      f'(set it via {roboflow_sync.CONFIG_FILE} under {data_root}).')
        roboflow_work = work / 'roboflow'
        if roboflow_work.exists(): shutil.rmtree(roboflow_work)
        data_yaml_path = roboflow_sync.download_dataset(cfg, args.roboflow_version, roboflow_work)
        parsed = roboflow_sync.read_yaml_names(data_yaml_path)
        merged_names = roboflow_sync.merge_class_names(names, parsed['names'])
        new_names = [n for n in parsed['names'] if n not in names]
        rf_index_map = {i: merged_names.index(n) for i, n in enumerate(parsed['names'])}

        for split, src in (('train', parsed['train']), ('val', parsed['val'])):
            if not src or not Path(src).is_dir():
                continue
            labels_src = Path(src).parent / 'labels'
            if not labels_src.is_dir():
                continue
            remapped_labels = roboflow_work / f'{split}_labels_remapped'
            roboflow_sync.remap_yolo_labels(labels_src, remapped_labels, rf_index_map)
            # Point a sibling 'labels' dir at the remapped labels, alongside the
            # original images dir, matching Ultralytics' images/labels layout.
            images_dst = roboflow_work / split / 'images'
            images_dst.parent.mkdir(parents=True, exist_ok=True)
            if images_dst.exists() or images_dst.is_symlink():
                images_dst.unlink() if images_dst.is_symlink() else shutil.rmtree(images_dst)
            images_dst.symlink_to(Path(src).resolve())
            labels_dst = roboflow_work / split / 'labels'
            if labels_dst.exists() or labels_dst.is_symlink():
                labels_dst.unlink() if labels_dst.is_symlink() else shutil.rmtree(labels_dst)
            labels_dst.symlink_to(remapped_labels.resolve())
            if split == 'train':
                roboflow_train_dir = images_dst
            else:
                roboflow_val_dir = images_dst
        names = merged_names
        print(f'roboflow dataset v{args.roboflow_version}: {len(parsed["names"])} classes, '
              f'{len(new_names)} new to the merged set: {new_names}')

    yaml, counts, total = assemble(data_root, work / 'dataset', names, teacher, args.min_corrections,
                                    skip_gate=roboflow_train_dir is not None)
    print(f'dataset: {total} corrections -> {counts}')
    print(f'final class count: {len(names)}')

    if roboflow_train_dir is not None:
        local_images_dir = (work / 'dataset' / 'images').resolve()
        val_dir = roboflow_val_dir if roboflow_val_dir is not None else local_images_dir
        yaml.write_text(
            f"path: {work / 'dataset'}\n"
            f"train:\n  - {local_images_dir}\n  - {roboflow_train_dir.resolve()}\n"
            f"val: {val_dir}\n"
            "names:\n" + ''.join(f"  {i}: {n}\n" for i, n in enumerate(names))
        )
    # Frozen backbone + small LR: adapt to these cameras, keep COCO knowledge.
    base.train(data=str(yaml), epochs=args.epochs, imgsz=640, device='mps', freeze=10, lr0=0.001,
               batch=8, project=str(work), name='run', exist_ok=True, verbose=False, plots=False)
    best = work / 'run' / 'weights' / 'best.pt'
    tuned = YOLO(str(best))
    exported = Path(tuned.export(format='coreml', nms=True, imgsz=640))
    # Into the app's data directory: the installed app looks there first, so
    # the tuned model takes effect on its next launch with no rebuild.
    target = data_root / 'models' / f'yolo11{args.size}-home.mlpackage'
    target.parent.mkdir(parents=True, exist_ok=True)
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
    print(f'exported {target}; quit and reopen ClearCam and it will use this model (Settings > Detection model stays on the same size).')


if __name__ == '__main__':
    main()
