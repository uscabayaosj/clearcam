"""Build a training set from public Roboflow Universe datasets, remapped onto
ClearCam's 8-class vocabulary, and upload it into your own Roboflow project so
training happens in Roboflow's cloud (this Mac runs out of memory training
locally on anything but the smallest datasets). Can also upload the owner's
own unlabelled camera frames (see script/collect_frames.py), teacher-labelled
in full since they carry no labels of their own.

Usage (training venv):
  <train-venv>/bin/python script/roboflow_dataset.py \
      --data "$HOME/Library/Application Support/ClearCam/Data" \
      --source marcu/stroller-vxfbx --source kpz3/stroller-tdpar \
      --source riley-zrx25/kids_adult-nqdo8 --source kicksquad/scooter-detect \
      --source whitera1313its-workspace/scooter-yhjgq \
      [--target-project clearcam-home] [--max-images 1500] [--teacher yolo11m.pt] \
      [--teacher-conf 0.5] [--workers 4] [--dry-run] [--work ~/.clearcam-rf-build]

Own-camera flow (collect -> select -> upload):
  python script/collect_frames.py collect --hours 12
  python script/collect_frames.py select --per-camera 150
  bash script/roboflow_dataset.sh --local ~/.clearcam-rf-build/own/selected \
      --teacher yolo11x.pt --teacher-conf 0.35

`--local DIR` (repeatable) uploads unlabelled frames from
<DIR>/train/images and <DIR>/valid/images (as produced by
script/collect_frames.py's `select` step) with the teacher labelling every
target class, since there is no source label to trust instead. <DIR>/holdout
is never read here - those frames are kept aside for script/compare_models.py
to judge a trained model, not for training. A local source needs no
--source at all, has no --max-images subsampling (there aren't many frames
to begin with), and requires a real --teacher (there's nothing else to label
the images with).

What it does, and why:
  1. Each --source is 'workspace/project' or 'workspace/project:version'; when
     no version is given, the source's latest published version is looked up
     (GET .../{workspace}/{project}?api_key=...) and used.
  2. Downloads each source's YOLOv11 export via utils.roboflow_sync
     (skips the download if it's already extracted under --work).
  3. Only 8 classes ever leave this tool: person, bicycle, car, truck, dog,
     stroller, child, scooter. Source class names are mapped onto them with
     utils.roboflow_sync.normalise_class (handles aliases, numeric prefixes,
     case); anything that doesn't map is dropped entirely.
  4. A deterministic, filename-hash-based subsample keeps each source to
     --max-images total (default 1500), respecting the source's own
     train/valid split with valid capped at ~15% of that budget - so a
     rerun with the same arguments always picks the same images, and a
     source with a huge valid/ split can't crowd out train.
  5. A COCO teacher (default yolo11m.pt; 'none' to skip) pseudo-labels the
     5 COCO classes in the vocabulary (person/bicycle/car/truck/dog) that a
     source doesn't already label itself, reusing
     script.train_from_corrections.augment_labels and its small-batch,
     one-image-at-a-time teacher loop (the Mac OOMs on bigger batches). A
     source's own labelled classes are trusted over the teacher, and a
     teacher 'person' box overlapping a 'child' label is skipped, same as
     the local trainer.
  6. Each prepared image is uploaded to --target-project (default: the
     configured project) with its own split ('train'/'valid'), a batch name
     'universe:<workspace>/<project>', and a Pascal VOC annotation built
     from its final YOLO labels. Filenames are prefixed with a short source
     slug so they stay unique across sources sharing a project.
  7. Uploads run concurrently (--workers) and retry transient errors
     (HTTP 429/5xx, timeouts) with backoff, up to 5 tries. An append-only
     ledger at <work>/uploaded.jsonl makes reruns skip what's already
     uploaded, and Ctrl-C just stops early - rerun the same command to
     resume.

Never logs the Roboflow API key: utils.roboflow_sync already redacts it out
of any HTTP error body before raising.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import roboflow_sync, static_filter
from script import train_from_corrections as tfc

TARGET_CLASSES = ['person', 'bicycle', 'car', 'truck', 'dog', 'stroller', 'child', 'scooter']
RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 504)


# --------------------------------------------------------------------- specs

def parse_source_spec(spec):
    """'workspace/project[:version]' -> (workspace, project, version_or_None)."""
    text = (spec or '').strip()
    version = None
    if ':' in text:
        text, _, version_part = text.rpartition(':')
        version_part = version_part.strip()
        if version_part:
            try:
                version = int(version_part)
            except ValueError:
                raise ValueError(f"invalid --source {spec!r}: version {version_part!r} is not an integer")
    if '/' not in text:
        raise ValueError(f"invalid --source {spec!r}: expected workspace/project[:version]")
    workspace, _, project = text.partition('/')
    workspace, project = workspace.strip(), project.strip()
    if not workspace or not project:
        raise ValueError(f"invalid --source {spec!r}: expected workspace/project[:version]")
    return workspace, project, version


def slugify(text):
    slug = re.sub(r'[^a-z0-9]+', '-', (text or '').lower()).strip('-')
    slug = re.sub(r'-{2,}', '-', slug)
    return slug or 'src'


def source_slug(workspace, project):
    return slugify(f'{workspace}-{project}')


def pick_latest_version(payload):
    """Pick the highest version number out of a Roboflow project GET response.

    Defensive on purpose: the documented shape is
    {"versions": [{"id": "workspace/project/3", ...}, ...]}, but this parses
    a 'version'/'num' integer or numeric-string field too, and tolerates
    items that are missing fields entirely (just skips them) rather than
    raising a KeyError partway through.
    """
    versions = payload.get('versions') if isinstance(payload, dict) else None
    best = None
    for item in versions or []:
        if not isinstance(item, dict):
            continue
        num = None
        raw_id = item.get('id')
        if isinstance(raw_id, str) and raw_id:
            tail = raw_id.rsplit('/', 1)[-1]
            if tail.isdigit():
                num = int(tail)
        if num is None:
            for key in ('version', 'num'):
                val = item.get(key)
                if isinstance(val, int):
                    num = val
                    break
                if isinstance(val, str) and val.strip().lstrip('v').isdigit():
                    num = int(val.strip().lstrip('v'))
                    break
        if num is not None and (best is None or num > best):
            best = num
    if best is None:
        raise roboflow_sync.RoboflowError('could not find a version in the Roboflow project response')
    return best


def fetch_latest_version(workspace, project, api_key, opener=None):
    opener = opener or roboflow_sync._default_opener
    url = f'https://api.roboflow.com/{workspace}/{project}?api_key={quote(api_key)}'
    request = urllib.request.Request(url, method='GET')
    try:
        resp = opener(request, timeout=30)
    except urllib.error.HTTPError as err:
        roboflow_sync._raise_http_error(err, api_key)
        return  # unreachable
    payload = roboflow_sync._read_json_response(resp)
    return pick_latest_version(payload)


# ---------------------------------------------------------------- subsampling

def deterministic_rank(names):
    return sorted(names, key=lambda n: hashlib.sha1(n.encode('utf-8')).hexdigest())


def deterministic_subsample(names, count):
    if count <= 0:
        return []
    return deterministic_rank(names)[:count]


def select_source_images(train_names, valid_names, max_images, val_fraction=0.15):
    """Deterministic per-source subsample capped at max_images total.

    Keeps the source's own train/valid split rather than remixing them: the
    valid portion is capped at ~val_fraction of max_images so a source with
    an unusually large valid/ split can't crowd out train; whatever budget
    remains goes to train, bounded by how many train images actually exist.
    Reruns with the same inputs always pick the same images (deterministic
    hash-based ranking), independent of directory iteration order.
    """
    val_cap = max(0, round(max_images * val_fraction))
    val_count = min(len(valid_names), val_cap)
    train_budget = max(0, max_images - val_count)
    train_count = min(len(train_names), train_budget)
    return deterministic_subsample(train_names, train_count), deterministic_subsample(valid_names, val_count)


# -------------------------------------------------------------- class mapping

def build_teacher_index_map(teacher_names):
    """Map each teacher (COCO) class index onto a TARGET_CLASSES index.

    The teacher only ever contributes person/bicycle/car/truck/dog boxes -
    it has no notion of stroller/child/scooter - so every other COCO class
    is simply absent from the returned map.
    """
    return roboflow_sync.build_index_map(teacher_names, TARGET_CLASSES)


def remap_lines(label_text, index_map):
    boxes = (roboflow_sync.yolo_box_line(line, index_map) for line in (label_text or '').splitlines())
    return [b for b in boxes if b is not None]


# --------------------------------------------------------------------- ledger

_ledger_lock = threading.Lock()


def ledger_path(work_dir):
    return Path(work_dir) / 'uploaded.jsonl'


def load_ledger(work_dir):
    path = ledger_path(work_dir)
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        name = row.get('filename')
        if name:
            done.add(name)
    return done


def append_ledger(work_dir, filename, roboflow_id):
    path = ledger_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock:
        with path.open('a') as stream:
            stream.write(json.dumps(dict(filename=filename, id=roboflow_id, time=time.time())) + '\n')


# --------------------------------------------------------------------- upload

def _is_transient(err):
    message = str(err)
    if any(f'HTTP {code}' in message for code in RETRYABLE_HTTP_CODES):
        return True
    return 'timed out' in message.lower()


def upload_with_retry(cfg, image_path, xml, split, batch, opener=None, max_tries=5, name=None):
    opener = opener or roboflow_sync._default_opener
    delay = 1.0
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return roboflow_sync.upload_image_and_annotation(cfg, image_path, xml, split=split, batch=batch, opener=opener,
                                                                   name=name)
        except roboflow_sync.RoboflowError as err:
            last_err = err
            if not _is_transient(err) or attempt == max_tries:
                raise
        except (urllib.error.URLError, socket.timeout, TimeoutError) as err:
            last_err = err
            if attempt == max_tries:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise last_err


# ------------------------------------------------------------- source prepare

def prepare_source(cfg, work_dir, workspace, project, version, max_images, teacher, teacher_index_map,
                    print_prefix=None, class_names=None, teacher_conf=0.5):
    """Download (if needed), subsample, remap and teacher-augment one source.

    Returns (records, counts): records is a list of dicts with path, lines
    (final YOLO label lines in TARGET_CLASSES indices), split, orig_name,
    width, height; counts is {class_name: box_count} for this source alone.
    """
    from PIL import Image
    print_prefix = print_prefix or f'{workspace}/{project}'
    dest_dir = Path(work_dir) / 'downloads' / f'{workspace}_{project}_v{version}'
    existing_yaml = list(dest_dir.rglob('data.yaml')) if dest_dir.is_dir() else []
    if existing_yaml:
        data_yaml_path = existing_yaml[0]
        print(f'{print_prefix}: using cached download at {dest_dir}', flush=True)
    else:
        cfg_src = dict(cfg, workspace=workspace, project=project)
        data_yaml_path = roboflow_sync.download_dataset(cfg_src, version, dest_dir)

    parsed = roboflow_sync.read_yaml_names(data_yaml_path)
    # Some exports carry garbled names in data.yaml (e.g. ['-', 'ksid adult - v5 ...']);
    # --class-names supplies the real ones, in label-index order.
    source_names = class_names or parsed['names']
    if class_names and len(class_names) != len(parsed['names']):
        raise SystemExit(f'{print_prefix}: --class-names gives {len(class_names)} names, '
                         f'the export has {len(parsed["names"])}: {parsed["names"]}')
    index_map = roboflow_sync.build_index_map(source_names, TARGET_CLASSES)
    print(f'{print_prefix}: classes {source_names} -> '
          f'{sorted({TARGET_CLASSES[i] for i in index_map.values()})}', flush=True)
    labelled_classes = set(index_map.values())

    train_dir, val_dir = tfc.export_split_dirs(data_yaml_path)
    train_images = tfc.list_images(train_dir) if train_dir else []
    if val_dir:
        val_images = tfc.list_images(val_dir)
    else:
        holdout = tfc.holdout_filenames([p.name for p in train_images], fraction=0.15)
        val_images = [p for p in train_images if p.name in holdout]
        train_images = [p for p in train_images if p.name not in holdout]

    train_sel, val_sel = select_source_images(
        [p.name for p in train_images], [p.name for p in val_images], max_images)
    train_by_name = {p.name: p for p in train_images}
    val_by_name = {p.name: p for p in val_images}
    selected = [(train_by_name[n], 'train') for n in train_sel] + [(val_by_name[n], 'valid') for n in val_sel]

    person_idx = TARGET_CLASSES.index('person') if 'person' in TARGET_CLASSES else None
    child_idx = TARGET_CLASSES.index('child') if 'child' in TARGET_CLASSES else None

    records = []
    counts = {name: 0 for name in TARGET_CLASSES}
    image_paths = [p for p, _ in selected]
    predictions = tfc.teacher_predict_batches(teacher, image_paths, conf=teacher_conf) if teacher is not None and image_paths else iter(())
    for n, (img_path, split) in enumerate(selected):
        result = next(predictions, None) if teacher is not None else None
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        existing_lines = remap_lines(label_path.read_text() if label_path.is_file() else '', index_map)
        with Image.open(img_path) as im:
            width, height = im.size
        lines = existing_lines
        if result is not None and len(result.boxes):
            teacher_boxes = [
                (teacher_index_map[int(c)], b, float(f))
                for b, c, f in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
                if int(c) in teacher_index_map
            ]
            if teacher_boxes:
                lines = tfc.augment_labels(existing_lines, teacher_boxes, labelled_classes, width, height,
                                            person_class=person_idx, child_class=child_idx)
        for line in lines:
            cls = int(line.split()[0])
            counts[TARGET_CLASSES[cls]] += 1
        records.append(dict(path=img_path, lines=lines, split=split, orig_name=img_path.name,
                             width=width, height=height))
        if (n + 1) % 500 == 0:
            print(f'{print_prefix}: prepared {n + 1}/{len(selected)}', flush=True)
    print(f'{print_prefix}: prepared {len(records)} images '
          f'({len(train_sel)} train, {len(val_sel)} valid)', flush=True)
    return records, counts


# --------------------------------------------------------------- local source

def local_source_images(dirpath):
    """(path, split) for every image under <dirpath>/train/images and
    <dirpath>/valid/images. <dirpath>/holdout is deliberately never read
    here - those frames are reserved for script/compare_models.py."""
    dirpath = Path(dirpath)
    pairs = []
    for split in ('train', 'valid'):
        d = dirpath / split / 'images'
        if not d.is_dir():
            continue
        for p in sorted(p for p in d.iterdir() if p.suffix.lower() in tfc.IMAGE_EXTS):
            pairs.append((p, split))
    return pairs


def _xyxy(line):
    cx, cy, w, h = (float(v) for v in line.split()[1:5])
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def merge_extra_labels(lines, extra, overlap=0.3):
    """Add hand-checked boxes [[class, x1, y1, x2, y2], ...] (normalised) to a
    frame's teacher lines. A teacher box of another class sitting on the same
    object (a pushchair called a bicycle, say) is dropped; people stay, since
    the person pushing overlaps the pushchair."""
    from utils.static_filter import iou
    added = []
    for name, x1, y1, x2, y2 in extra:
        if name not in TARGET_CLASSES:
            raise ValueError(f'extra label class {name!r} is not one of {TARGET_CLASSES}')
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        added.append((TARGET_CLASSES.index(name), (x1, y1, x2, y2)))
    keep_classes = {TARGET_CLASSES.index('person'), TARGET_CLASSES.index('child')}
    kept = [l for l in lines
            if int(l.split()[0]) in keep_classes
            or not any(int(l.split()[0]) != c and iou(_xyxy(l), b) > overlap for c, b in added)]
    for c, (x1, y1, x2, y2) in added:
        kept.append(f'{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
    return kept


def prepare_local(dirpath, teacher, teacher_index_map, teacher_conf=0.5, print_prefix=None, ignore_areas=None,
                  extra_labels=None):
    """Teacher-label every image under a local (own-camera) source directory.

    These frames carry no labels of their own (collect_frames.py's `select`
    step writes bare images), so the teacher labels every target class the
    normal partial-label path would otherwise trust to a source's own
    annotations for - labelled_classes is passed empty. Unlike a Roboflow
    Universe source, there is no --max-images subsampling: there typically
    aren't many own-camera frames to begin with.

    extra_labels ({file name: [[class, x1, y1, x2, y2], ...]}, normalised)
    adds hand-checked boxes for classes the teacher can't see (pushchairs).
    """
    from PIL import Image
    dirpath = Path(dirpath)
    print_prefix = print_prefix or f'local:{dirpath.name}'
    pairs = local_source_images(dirpath)
    person_idx = TARGET_CLASSES.index('person')
    child_idx = TARGET_CLASSES.index('child')
    batch = f'own:{dirpath.name}'

    records = []
    counts = {name: 0 for name in TARGET_CLASSES}
    image_paths = [p for p, _ in pairs]
    predictions = tfc.teacher_predict_batches(teacher, image_paths, conf=teacher_conf) if image_paths else iter(())
    for n, (img_path, split) in enumerate(pairs):
        result = next(predictions, None)
        with Image.open(img_path) as im:
            width, height = im.size
        lines = []
        if result is not None and len(result.boxes):
            teacher_boxes = [
                (teacher_index_map[int(c)], b, float(f))
                for b, c, f in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
                if int(c) in teacher_index_map
            ]
            if teacher_boxes:
                lines = tfc.augment_labels([], teacher_boxes, set(), width, height,
                                            person_class=person_idx, child_class=child_idx)
        checked = set()
        if extra_labels and img_path.name in extra_labels:
            before = set(lines)
            lines = merge_extra_labels(lines, extra_labels[img_path.name])
            checked = set(lines) - before
        for line in lines:
            cls = int(line.split()[0])
            counts[TARGET_CLASSES[cls]] += 1
        records.append(dict(path=img_path, lines=lines, split=split, orig_name=img_path.name,
                             width=width, height=height, final_name=img_path.name, batch=batch,
                             checked=checked))
        if (n + 1) % 500 == 0:
            print(f'{print_prefix}: prepared {n + 1}/{len(pairs)}', flush=True)
    records, counts, dropped = drop_static_labels(records, ignore_areas)
    print(f'{print_prefix}: prepared {len(records)} images from local frames '
          f'({dropped} boxes dropped as scenery or in ignore areas)', flush=True)
    return records, counts


def drop_static_labels(records, ignore_areas=None):
    """Remove movable-class labels that sit at the same spot across a camera's
    frames (see utils/static_filter.py). Returns (records, counts, dropped)."""
    from utils import static_filter
    def box(line):
        cls, cx, cy, w, h = line.split()[:5]
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        return TARGET_CLASSES[int(cls)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    frames = [(static_filter.camera_of(r['final_name']), [box(l) for l in r['lines']]) for r in records]
    masks = static_filter.static_mask(frames)
    # Hand-marked areas where the teacher keeps seeing something that isn't
    # there (e.g. bins called cars): drop those labels so the frames teach
    # the model the opposite.
    for (camera, boxes), mask in zip(frames, masks):
        for j, (name, b) in enumerate(boxes):
            if static_filter.ignored(camera, name, b, ignore_areas or {}):
                mask[j] = False
    # Boxes checked by eye are never scenery: a pushchair that pauses sits at
    # one spot across frames, and dropping it would teach "not a pushchair".
    for rec, mask in zip(records, masks):
        for j, line in enumerate(rec['lines']):
            if line in rec.get('checked', ()):
                mask[j] = True
    counts = {name: 0 for name in TARGET_CLASSES}
    dropped = 0
    for rec, mask in zip(records, masks):
        rec['lines'] = [l for l, keep in zip(rec['lines'], mask) if keep]
        dropped += mask.count(False)
        for l in rec['lines']:
            counts[TARGET_CLASSES[int(l.split()[0])]] += 1
    return records, counts, dropped


# -------------------------------------------------------------------- driver

def format_counts(counts):
    return ', '.join(f'{k}={v}' for k, v in counts.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='ClearCam Data directory')
    parser.add_argument('--source', action='append', default=[], dest='sources',
                         help="Repeatable. 'workspace/project' or 'workspace/project:version'; "
                              "omit the version to use the source's latest.")
    parser.add_argument('--local', action='append', default=[], dest='locals',
                         help="Repeatable. A local (own-camera) source directory with train/images and "
                              "valid/images (holdout/images, if present, is ignored). Needs a real "
                              "--teacher: these frames have no labels of their own.")
    parser.add_argument('--max-images', type=int, default=1500, help='per Universe source (default 1500); '
                                                                       'local sources are never subsampled')
    parser.add_argument('--teacher', default='yolo11m.pt', help="'none' to skip teacher pseudo-labels "
                                                                  "(not allowed together with --local)")
    parser.add_argument('--ignore', action='append', default=[],
                         help="Repeatable, own-camera sources only: 'camera=x1,y1,x2,y2[:class,...]' (0-1 coords). "
                              "Teacher labels centred there are dropped (e.g. bins it calls cars).")
    parser.add_argument('--extra-labels', default=None,
                        help="JSON {file name: [[class, x1, y1, x2, y2], ...]} (normalised) of hand-checked "
                             "boxes added to --local frames, for classes the teacher can't label (stroller)")
    parser.add_argument('--teacher-conf', type=float, default=0.5, help='teacher confidence threshold (default 0.5)')
    parser.add_argument('--target-project', default=None, help='default: the configured project')
    parser.add_argument('--workers', type=int, default=4, help='concurrent uploads (default 4)')
    parser.add_argument('--class-names', action='append', default=[],
                         help="Repeatable. 'workspace/project=name0,name1,...' overrides a source's "
                              "class names (in label-index order) when its export's are wrong.")
    parser.add_argument('--dry-run', action='store_true', help='prepare locally, upload nothing')
    parser.add_argument('--work', default=str(Path.home() / '.clearcam-rf-build'),
                         help='scratch dir for downloads/ledger (default ~/.clearcam-rf-build)')
    args = parser.parse_args()

    if not args.sources and not args.locals:
        sys.exit('at least one --source workspace/project[:version] or --local DIR is required')
    if args.locals and (args.teacher or 'none').strip().lower() == 'none':
        sys.exit('--teacher none cannot be used with --local: local frames have no labels of their own '
                  'for the teacher to fall back from')

    data_root = Path(args.data).expanduser()
    work_dir = Path(args.work).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = roboflow_sync.load_config(data_root)
    if not cfg.get('api_key'):
        sys.exit(f'Roboflow API key not configured (see {roboflow_sync.CONFIG_FILE} under {data_root}).')
    target_project = args.target_project or cfg.get('project')
    if not target_project and not args.dry_run:
        sys.exit('no --target-project given and none configured; set one, or pass --dry-run to skip uploads.')

    name_overrides = {}
    for raw in args.class_names:
        key, sep, names = raw.partition('=')
        if not sep or '/' not in key or not names.strip():
            sys.exit(f"invalid --class-names {raw!r}: expected workspace/project=name0,name1,...")
        name_overrides[key.strip()] = [n.strip() for n in names.split(',')]

    specs = []
    for raw in args.sources:
        try:
            workspace, project, version = parse_source_spec(raw)
        except ValueError as err:
            sys.exit(str(err))
        if version is None:
            print(f'{workspace}/{project}: looking up latest version...', flush=True)
            version = fetch_latest_version(workspace, project, cfg['api_key'])
            print(f'{workspace}/{project}: using version {version}', flush=True)
        specs.append((workspace, project, version))

    teacher = None
    teacher_index_map = {}
    if (args.teacher or 'none').strip().lower() != 'none':
        from ultralytics import YOLO
        teacher = YOLO(args.teacher)
        teacher_names = [teacher.names[i] for i in range(len(teacher.names))]
        teacher_index_map = build_teacher_index_map(teacher_names)

    all_records = []
    totals = {name: 0 for name in TARGET_CLASSES}
    for workspace, project, version in specs:
        slug = source_slug(workspace, project)
        prefix = f'{workspace}/{project} v{version}'
        records, counts = prepare_source(cfg, work_dir, workspace, project, version, args.max_images,
                                          teacher, teacher_index_map, print_prefix=prefix,
                                          class_names=name_overrides.get(f'{workspace}/{project}'),
                                          teacher_conf=args.teacher_conf)
        print(f'{prefix}: box counts {format_counts(counts)}', flush=True)
        for k, v in counts.items():
            totals[k] += v
        batch = f'universe:{workspace}/{project}'
        for rec in records:
            rec['final_name'] = f"{slug}_{rec['orig_name']}"
            rec['batch'] = batch
        all_records.extend(records)

    try:
        ignore_areas = static_filter.parse_ignore(args.ignore)
    except ValueError as err:
        sys.exit(str(err))
    extra_labels = json.loads(Path(args.extra_labels).expanduser().read_text()) if args.extra_labels else None
    for raw_dir in args.locals:
        dirpath = Path(raw_dir).expanduser()
        prefix = f'local:{dirpath.name}'
        records, counts = prepare_local(dirpath, teacher, teacher_index_map,
                                         teacher_conf=args.teacher_conf, print_prefix=prefix,
                                         ignore_areas=ignore_areas, extra_labels=extra_labels)
        print(f'{prefix}: box counts {format_counts(counts)}', flush=True)
        for k, v in counts.items():
            totals[k] += v
        all_records.extend(records)

    print(f'totals across all sources: {format_counts(totals)}')
    print(f'{len(all_records)} images prepared total', flush=True)

    if args.dry_run:
        print('dry run: nothing uploaded.')
        return

    cfg_upload = dict(cfg, project=target_project)
    already = load_ledger(work_dir)
    todo = [r for r in all_records if r['final_name'] not in already]
    print(f'{len(all_records) - len(todo)} already uploaded (ledger), {len(todo)} to upload', flush=True)

    def upload_one(rec):
        xml = roboflow_sync.voc_xml_from_yolo_lines(rec['lines'], TARGET_CLASSES, rec['final_name'],
                                                     rec['width'], rec['height'])
        return upload_with_retry(cfg_upload, rec['path'], xml, rec['split'], rec['batch'], name=rec['final_name'])

    start = time.time()
    done = 0
    failed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(upload_one, rec): rec for rec in todo}
            for future in concurrent.futures.as_completed(futures):
                rec = futures[future]
                try:
                    result = future.result()
                except Exception as err:  # noqa: BLE001 - one failed upload must not abort the batch
                    failed += 1
                    print(f'upload failed for {rec["final_name"]}: {err}', flush=True)
                    continue
                append_ledger(work_dir, rec['final_name'], (result or {}).get('id'))
                done += 1
                if done % 100 == 0:
                    elapsed = max(0.001, time.time() - start)
                    rate = done / elapsed
                    remaining = len(todo) - done
                    eta_min = (remaining / rate / 60) if rate > 0 else float('inf')
                    print(f'uploaded {done}/{len(todo)} ({rate:.1f}/s, ETA {eta_min:.1f} min)', flush=True)
    except KeyboardInterrupt:
        print(f'\ninterrupted: {done} uploaded this run ({failed} failed). '
              'Rerun the same command to resume - already-uploaded images are skipped via the ledger.', flush=True)
        return

    print(f'done: {done} uploaded, {failed} failed this run.')
    print(f'totals across all sources: {format_counts(totals)}')
    print('Generate a version in Roboflow, train YOLO11 object detection, then: '
          'bash script/roboflow_model.sh <version>')


if __name__ == '__main__':
    main()
