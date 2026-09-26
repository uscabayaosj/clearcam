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
  3. Optionally merges in one or more external Roboflow datasets
     (--roboflow-dataset / --roboflow-version), remapping their classes onto
     the ClearCam vocabulary and pseudo-labelling COCO classes those datasets
     don't label themselves, so the student doesn't unlearn "car" just
     because a doorbell dataset never bothered to box one.
  4. Fine-tunes with the backbone frozen and a small learning rate, so the
     model adapts to these cameras without forgetting the COCO classes.
  5. Runs a safety gate against the stock model before installing: a tuned
     model whose COCO mAP has collapsed is left in the work dir instead.
  6. Exports Core ML as models/yolo11<size>-home.mlpackage; the engine prefers
     a -home package over the stock one of the same size.
Needs the export venv (ultralytics, torch, coremltools). Ultralytics is AGPL;
this is for personal installs unless that is resolved for distribution.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import roboflow_sync
COCO = [l.strip() for l in (ROOT / 'models' / 'coco.names').read_text().splitlines() if l.strip()] if (ROOT / 'models' / 'coco.names').exists() else None

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def to_yolo_line(cls, box, width, height):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def _parse_yolo_line(line, width, height):
    parts = line.split()
    cls = int(parts[0])
    cx, cy, w, h = (float(x) for x in parts[1:5])
    x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
    x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
    return cls, [x1, y1, x2, y2]


def augment_labels(existing_lines, teacher_boxes, labelled_classes, width, height,
                    person_class=None, child_class=None):
    """Add teacher pseudo-labels for COCO classes a dataset doesn't label itself.

    existing_lines: YOLO-normalised text lines already in the label file.
    teacher_boxes: list of (cls, [x1, y1, x2, y2] pixel box, conf) from the teacher.
    labelled_classes: merged class indices this dataset labels itself - a
      teacher box for one of these classes is skipped, because the dataset's
      own (lack of a) label for it is trusted, not the teacher's guess.
    person_class/child_class: merged indices of 'person'/'child', if both
      exist in the vocabulary. A teacher 'person' box is also skipped when it
      overlaps (IoU>0.3) an existing 'child' label, so a child doesn't get a
      second, contradictory 'person' box drawn over it.
    Returns existing_lines plus one appended line per accepted teacher box.
    Never mutates existing_lines.
    """
    existing = [_parse_yolo_line(l, width, height) for l in existing_lines if l.strip()]
    existing_boxes = [box for _cls, box in existing]
    child_boxes = [box for cls, box in existing if child_class is not None and cls == child_class]
    added = []
    for cls, box, _conf in teacher_boxes:
        if cls in labelled_classes:
            continue
        if any(iou(box, eb) > 0.5 for eb in existing_boxes):
            continue
        if person_class is not None and cls == person_class and any(iou(box, cb) > 0.3 for cb in child_boxes):
            continue
        added.append(to_yolo_line(cls, box, width, height))
    return list(existing_lines) + added


def parse_roboflow_spec(spec, default_workspace):
    """Parse '--roboflow-dataset' SPEC: 'project:version' or 'workspace/project:version'."""
    spec = (spec or '').strip()
    if ':' not in spec:
        raise ValueError(f"invalid --roboflow-dataset spec {spec!r}: expected project:version or workspace/project:version")
    proj_part, _, version_part = spec.rpartition(':')
    proj_part = proj_part.strip()
    try:
        version = int(version_part.strip())
    except ValueError:
        raise ValueError(f"invalid --roboflow-dataset spec {spec!r}: version {version_part!r} is not an integer")
    if '/' in proj_part:
        workspace, _, project = proj_part.partition('/')
    else:
        workspace, project = default_workspace, proj_part
    workspace, project = workspace.strip(), project.strip()
    if not project:
        raise ValueError(f"invalid --roboflow-dataset spec {spec!r}: missing project")
    if not workspace:
        raise ValueError(f"invalid --roboflow-dataset spec {spec!r}: no workspace given and none configured")
    return workspace, project, version


def build_merged_names(base_names, new_classes):
    """The fixed training vocabulary: base_names (COCO), plus each name in
    new_classes appended in the order given (skipping blanks and names
    already present). Unlike base_names, new_classes indices are decided
    once, up front - not by what any particular dataset happens to contain -
    so 'stroller','child','scooter' always land at the same indices whether
    or not a given dataset uses all three.
    """
    merged = list(base_names)
    lower_set = {n.lower() for n in merged}
    for name in new_classes:
        name = (name or '').strip()
        if not name or name.lower() in lower_set:
            continue
        merged.append(name)
        lower_set.add(name.lower())
    return merged


def build_index_map_with_drops(src_names, merged_names, drop_classes=None):
    """Like roboflow_sync.build_index_map, but a source class normalising to
    a name in drop_classes (case-insensitive) is excluded even if it would
    otherwise match a class in merged_names - --drop-classes is applied
    before COCO/new-class matching.
    """
    drop_set = {d.strip().lower() for d in (drop_classes or []) if d.strip()}
    index_map = roboflow_sync.build_index_map(src_names, merged_names)
    if not drop_set:
        return index_map
    out = {}
    for i, name in enumerate(src_names):
        norm = roboflow_sync.normalise_class(name, merged_names)
        if norm.lower() in drop_set:
            continue
        if i in index_map:
            out[i] = index_map[i]
    return out


def nothing_new_to_learn(new_class_names, disagreements):
    """True when an external-data run would only reinforce the stock model.

    new_class_names: names added to the vocabulary by any supplied dataset.
    disagreements: count of local corrections that disagree with the model.
    """
    return not new_class_names and disagreements < 1


def balance_repeat_factor(local_count, external_count, target_fraction=0.10, min_factor=1, max_factor=20):
    """How many times to repeat the local corrections images in the train list.

    Solves for factor such that local*factor is ~target_fraction of the
    combined (local*factor + external) train set, clamped to
    [min_factor, max_factor].
    """
    if local_count <= 0 or external_count <= 0:
        return min_factor
    factor = (target_fraction * external_count) / (local_count * (1 - target_fraction))
    factor = round(factor)
    return max(min_factor, min(max_factor, factor))


def holdout_filenames(filenames, fraction=0.10):
    """Deterministically pick ~fraction of filenames as a validation holdout.

    Ranks filenames by the hash of their name (not by content or mtime) so
    the same holdout is chosen on every run, independent of directory
    iteration order.
    """
    names = list(filenames)
    if not names:
        return set()
    ranked = sorted(names, key=lambda f: hashlib.sha1(f.encode('utf-8')).hexdigest())
    n = max(1, round(len(ranked) * fraction))
    return set(ranked[:n])


def teacher_predict_batches(teacher, paths, batch=16, conf=0.5, device='mps'):
    """Yield one ultralytics Result per path in paths, running the teacher in
    small batches so a large dataset never gets loaded into memory at once.

    Given the whole list at once, ultralytics treats it as ONE batch and
    decodes every image up front - a 6,000-image dataset got the process
    killed on this Mac. Batches of 16 keep memory bounded while still using
    the GPU (device='mps') for throughput.
    """
    paths = list(paths)
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        yield from teacher.predict([str(x) for x in chunk], conf=conf, verbose=False, batch=len(chunk), device=device)


def list_images(d):
    d = Path(d)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _symlink_files(paths, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        link = dest_dir / p.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(p.resolve())


def assemble(data_root, out_dir, names, teacher=None, min_corrections=20, skip_gate=False):
    from PIL import Image
    store = data_root / 'corrections'
    rows = [json.loads(l) for l in (store / 'corrections.jsonl').read_text().splitlines() if l.strip()] if (store / 'corrections.jsonl').exists() else []
    disagreements = sum(1 for r in rows if r['verdict'] != 'confirm')
    # "Looks right" alone is the model grading its own homework: only verdicts
    # that disagree with it carry information the fine-tune can learn from.
    # When external data is supplied there is data to train on regardless of
    # the local count, so the local-corrections gate is informational only.
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


def export_split_dirs(data_yaml_path):
    """(train_images_dir, val_images_dir or None) inside a Roboflow export.

    Located on disk rather than trusted from data.yaml: Roboflow writes paths
    like '../train/images' that are one directory off from the yaml itself.
    """
    root = Path(data_yaml_path).parent
    def first(*names):
        for name in names:
            d = root / name / 'images'
            if d.is_dir() and list_images(d): return d
        return None
    return first('train'), first('valid', 'val')


def _prepare_external_dataset(key, data_yaml_path, index_map, out_root, teacher, print_prefix,
                              person_class=None, child_class=None):
    """Remap an external dataset's labels into the merged vocabulary, add
    teacher pseudo-labels for COCO classes it leaves unlabelled, and lay it out
    as out_root/key/{train,val}/{images,labels} — the images/labels sibling
    layout the trainer needs to pair each image with its label file.

    Never writes inside the download (out_root must be elsewhere), and handles
    teacher results one image at a time: holding them all would keep every
    decoded frame of a 10k-image dataset in memory.
    """
    labelled_classes = set(index_map.values())
    dataset_dir = Path(out_root) / key
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    def remap(label_file):
        if not label_file.is_file(): return []
        out = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            try: cls = int(parts[0])
            except ValueError: continue
            if cls not in index_map: continue
            parts[0] = str(index_map[cls])
            out.append(' '.join(parts[:5]))
        return out

    def augment_split(image_paths, split_name):
        images_out = dataset_dir / split_name / 'images'
        labels_out = dataset_dir / split_name / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        n_images = n_orig = n_teacher = 0
        predictions = teacher_predict_batches(teacher, image_paths) if teacher is not None and image_paths else iter(())
        for img_path in image_paths:
            result = next(predictions, None) if teacher is not None else None
            link = images_out / img_path.name
            if not link.exists(): link.symlink_to(img_path.resolve())
            n_images += 1
            existing = remap(img_path.parent.parent / 'labels' / (img_path.stem + '.txt'))
            n_orig += len(existing)
            lines = existing
            if result is not None and len(result.boxes):
                height, width = result.orig_shape[:2]
                teacher_boxes = [(int(c), b, float(f)) for b, c, f in zip(
                    result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]
                lines = augment_labels(existing, teacher_boxes, labelled_classes, width, height,
                                       person_class=person_class, child_class=child_class)
                n_teacher += len(lines) - len(existing)
            (labels_out / (img_path.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
            if n_images % 1000 == 0:
                print(f'{print_prefix} {split_name}: {n_images}/{len(image_paths)} prepared', flush=True)
        print(f'{print_prefix} {split_name}: {n_images} images, {n_orig} original boxes, {n_teacher} teacher boxes added', flush=True)
        return images_out if n_images else None

    train_src, val_src = export_split_dirs(data_yaml_path)
    train_images = list_images(train_src) if train_src else []
    if val_src:
        val_images = list_images(val_src)
    else:
        # No valid split supplied: carve a deterministic 10% holdout from train.
        holdout = holdout_filenames([x.name for x in train_images], fraction=0.10)
        val_images = [x for x in train_images if x.name in holdout]
        train_images = [x for x in train_images if x.name not in holdout]
    if not train_images:
        print(f'{print_prefix}: no training images found in the export', flush=True)
    return augment_split(train_images, 'train'), augment_split(val_images, 'val')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='ClearCam Data directory')
    parser.add_argument('--size', default='s', choices=['n', 's'])
    parser.add_argument('--epochs', type=int, default=None, help='default 20 with only local corrections, 15 when external datasets are used')
    parser.add_argument('--teacher', default='yolo11m.pt', help="'none' to skip teacher relabeling")
    parser.add_argument('--out', default=None)
    parser.add_argument('--min-corrections', type=int, default=20)
    parser.add_argument('--roboflow-version', type=int, default=None,
                         help='Shorthand for --roboflow-dataset using the configured project.')
    parser.add_argument('--roboflow-dataset', action='append', default=[],
                         help="Repeatable. 'project:version' or 'workspace/project:version'; "
                              "workspace defaults to the configured one.")
    parser.add_argument('--drop-classes', default='', help='Comma-separated class names to exclude entirely.')
    parser.add_argument('--new-classes', default='stroller,child,scooter',
                         help='Comma-separated class names (besides COCO) to keep, appended in this order. '
                              'Any other name external datasets use is dropped.')
    args = parser.parse_args()
    from ultralytics import YOLO
    data_root = Path(args.data).expanduser()
    work = Path(args.out) if args.out else data_root / 'training' / f'yolo11{args.size}-home'
    if work.exists(): shutil.rmtree(work)
    base = YOLO(f'yolo11{args.size}.pt')
    names = [base.names[i] for i in range(len(base.names))]
    teacher = None if args.teacher == 'none' else YOLO(args.teacher)
    drop_classes = [c for c in (args.drop_classes or '').split(',') if c.strip()]
    new_classes = [c for c in (args.new_classes or '').split(',') if c.strip()]

    dataset_requests = []  # list of (workspace, project, version)
    if args.roboflow_version is not None or args.roboflow_dataset:
        cfg = roboflow_sync.load_config(data_root)
        missing = [k for k in ('api_key',) if not cfg.get(k)]
        if missing:
            sys.exit(f'--roboflow-version/--roboflow-dataset need Roboflow config: missing {", ".join(missing)} '
                      f'(set it via {roboflow_sync.CONFIG_FILE} under {data_root}).')
        if args.roboflow_version is not None:
            if not cfg.get('workspace') or not cfg.get('project'):
                sys.exit('--roboflow-version needs a configured workspace and project '
                         f'(set it via {roboflow_sync.CONFIG_FILE} under {data_root}).')
            dataset_requests.append((cfg['workspace'], cfg['project'], args.roboflow_version))
        for spec in args.roboflow_dataset:
            try:
                dataset_requests.append(parse_roboflow_spec(spec, cfg.get('workspace', '')))
            except ValueError as err:
                sys.exit(str(err))

    downloaded = []  # list of dicts: workspace, project, version, key, parsed
    if dataset_requests:
        rf_root = work / 'roboflow'
        for workspace, project, version in dataset_requests:
            key = f'{project}_v{version}'
            dest_dir = rf_root / key
            if dest_dir.exists(): shutil.rmtree(dest_dir)
            cfg_i = dict(cfg)
            cfg_i['workspace'], cfg_i['project'] = workspace, project
            data_yaml_path = roboflow_sync.download_dataset(cfg_i, version, dest_dir)
            parsed = roboflow_sync.read_yaml_names(data_yaml_path)
            downloaded.append(dict(workspace=workspace, project=project, version=version, key=key, parsed=parsed,
                                   data_yaml=data_yaml_path))

        # Fixed vocabulary: COCO + the configured new classes, always at the
        # same indices. Anything an external dataset labels that isn't COCO
        # or in --new-classes is dropped (see build_index_map_with_drops).
        names = build_merged_names(names, new_classes)
        base_class_count = len(base.names)

    # Index maps are cheap to build, so compute them (and check the
    # nothing-new gate) before doing any of the expensive per-image teacher
    # pseudo-labeling work in _prepare_external_dataset.
    index_maps = {}
    new_names_used = set()
    if downloaded:
        for d in downloaded:
            index_map = build_index_map_with_drops(d['parsed']['names'], names, drop_classes=drop_classes)
            index_maps[d['key']] = index_map
            new_names_used |= {i for i in index_map.values() if i >= base_class_count}
        new_names = [names[i] for i in sorted(new_names_used)]

    total_disagreements = None  # computed by assemble(); check the gate up front instead
    if dataset_requests:
        store = data_root / 'corrections'
        rows = [json.loads(l) for l in (store / 'corrections.jsonl').read_text().splitlines() if l.strip()] if (store / 'corrections.jsonl').exists() else []
        total_disagreements = sum(1 for r in rows if r['verdict'] != 'confirm')
        if nothing_new_to_learn(new_names, total_disagreements):
            sys.exit('Nothing new to learn: the datasets only contain classes the detector already knows '
                      'and no corrections disagree with it.')

    train_dirs, val_dirs = [], []
    if downloaded:
        rf_work = work / 'external'   # never inside work/'roboflow', where the downloads live
        person_idx = names.index('person') if 'person' in names else None
        child_idx = names.index('child') if 'child' in names else None
        for d in downloaded:
            train_dir, val_dir = _prepare_external_dataset(
                d['key'], d['data_yaml'], index_maps[d['key']], rf_work, teacher, f"{d['project']} v{d['version']}",
                person_class=person_idx, child_class=child_idx)
            if train_dir is not None: train_dirs.append(train_dir)
            if val_dir is not None: val_dirs.append(val_dir)
        print(f'roboflow datasets: {len(downloaded)} merged, {len(new_names)} new classes actually used: {new_names}')

    yaml, counts, total = assemble(data_root, work / 'dataset', names, teacher, args.min_corrections,
                                    skip_gate=bool(downloaded))
    print(f'dataset: {total} corrections -> {counts}')
    print(f'final class count: {len(names)}')

    external_present = bool(downloaded)
    if external_present:
        local_images_dir = (work / 'dataset' / 'images').resolve()
        local_count = len(list_images(local_images_dir))
        external_count = sum(len(list_images(d)) for d in train_dirs)
        repeat = balance_repeat_factor(local_count, external_count)
        print(f'balance: repeating {local_count} local images x{repeat} against {external_count} external train images')
        train_list = [str(d.resolve()) for d in train_dirs] + [str(local_images_dir)] * repeat
        val_list = [str(d.resolve()) for d in val_dirs] if val_dirs else [str(local_images_dir)]
        yaml.write_text(
            f"path: {work / 'dataset'}\n"
            "train:\n" + ''.join(f"  - {p}\n" for p in train_list) +
            "val:\n" + ''.join(f"  - {p}\n" for p in val_list) +
            "names:\n" + ''.join(f"  {i}: {n}\n" for i, n in enumerate(names))
        )

    epochs = args.epochs if args.epochs is not None else (15 if external_present else 20)
    # Frozen backbone + small LR: adapt to these cameras, keep COCO knowledge.
    # A bigger, more varied external-data mix gets a schedule suited to it:
    # slightly higher LR, less mosaic tail, earlier stopping on plateau.
    train_kwargs = dict(data=str(yaml), epochs=epochs, imgsz=640, device='mps', freeze=10,
                         project=str(work), name='run', exist_ok=True, verbose=False, plots=False)
    if external_present:
        train_kwargs.update(lr0=0.002, close_mosaic=3, patience=5, cache='disk', batch=16, workers=4)
    else:
        train_kwargs.update(lr0=0.001, batch=8)
    base.train(**train_kwargs)
    best = work / 'run' / 'weights' / 'best.pt'
    tuned = YOLO(str(best))

    target = data_root / 'models' / f'yolo11{args.size}-home.mlpackage'
    install_ok = True
    if external_present and val_dirs:
        # Safety gate: a tuned model that has forgotten COCO is worse than
        # the stock one, even if it nails the new/local classes. Compare
        # COCO-only mAP50-95 on the same combined val set.
        combined_val_yaml = work / 'val_coco.yaml'
        coco_names_in_merge = [n for n in names[:len(base.names)]]
        combined_val_yaml.write_text(
            "path: " + str(work) + "\n"
            "train:\n  - " + str(val_dirs[0].resolve()) + "\n"
            "val:\n" + ''.join(f"  - {p.resolve()}\n" for p in val_dirs) +
            "names:\n" + ''.join(f"  {i}: {n}\n" for i, n in enumerate(names))
        )
        try:
            stock_metrics = base.val(data=str(combined_val_yaml), imgsz=640, device='mps', verbose=False, plots=False)
            tuned_metrics = tuned.val(data=str(combined_val_yaml), imgsz=640, device='mps', verbose=False, plots=False)
            stock_maps = stock_metrics.box.maps
            tuned_maps = tuned_metrics.box.maps
            coco_idx = list(range(len(coco_names_in_merge)))
            stock_coco_map = sum(stock_maps[i] for i in coco_idx if i < len(stock_maps)) / max(1, len(coco_idx))
            tuned_coco_map = sum(tuned_maps[i] for i in coco_idx if i < len(tuned_maps)) / max(1, len(coco_idx))
            if stock_coco_map > 0 and (stock_coco_map - tuned_coco_map) / stock_coco_map > 0.10:
                install_ok = False
                print(f'safety gate: tuned COCO mAP50-95 {tuned_coco_map:.4f} vs stock {stock_coco_map:.4f} '
                      f'is a {(stock_coco_map - tuned_coco_map) / stock_coco_map:.1%} relative drop (>10%); not installing.')
            else:
                print(f'safety gate: tuned COCO mAP50-95 {tuned_coco_map:.4f} vs stock {stock_coco_map:.4f} - OK')
            for cls_name in ('person', 'car', 'truck', 'bicycle', 'dog', 'stroller', 'child', 'scooter'):
                if cls_name in names:
                    idx = names.index(cls_name)
                    val = tuned_maps[idx] if idx < len(tuned_maps) else None
                    if val is not None:
                        print(f'  per-class mAP50(-95) {cls_name}: {val:.4f}')
        except Exception as err:  # noqa: BLE001 - a failed safety eval must not block reporting; it blocks install
            install_ok = False
            print(f'safety gate: could not evaluate ({err}); not installing.')

    exported = Path(tuned.export(format='coreml', nms=True, imgsz=640))
    if not install_ok:
        print(f'tuned model left at {exported}; not installed into {target} (see safety gate message above).')
    else:
        # Into the app's data directory: the installed app looks there first, so
        # the tuned model takes effect on its next launch with no rebuild.
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
