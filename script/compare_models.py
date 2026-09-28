"""Compare a candidate detector against the stock detector on the owner's own
holdout frames (script/collect_frames.py's `select` step sets these aside,
unlabelled and never uploaded).

Usage (training venv):
  <train-venv>/bin/python script/compare_models.py \
      --candidate "$HOME/Library/Application Support/ClearCam/Data/training/roboflow-models/clearcam-home-v4/weights.pt" \
      [--holdout ~/.clearcam-rf-build/own/selected/holdout/images] \
      [--stock yolo11s.pt] [--reference yolo11x.pt] [--conf 0.4] \
      [--sheet compare.jpg]

Why: the holdout frames have no ground-truth labels (collecting labels for
them would defeat the point of a cheap own-camera check), so a big reference
model's own detections stand in for ground truth. The stock and candidate
detectors are then scored against those reference boxes the same way, class
by class, restricted to the 8 classes ClearCam cares about
(`script.roboflow_dataset.TARGET_CLASSES`) by matching class NAME - the
candidate uses its own (possibly reordered) class indices, while stock/
reference use plain COCO names.

Detection counts/recall/precision, matching (IoU>=0.5, greedy by confidence,
same class), and the verdict rule are all pure functions with no model or
file I/O, so they're unit-testable with fake boxes - see
test/test_compare_models.py. The actual model inference is the only part
that touches ultralytics/PIL, and it runs one image at a time on device='mps'
since this Mac OOMs on larger batches.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from script import train_from_corrections as tfc
from script.roboflow_dataset import TARGET_CLASSES
from utils import static_filter

IOU_THRESHOLD = 0.5
REFERENCE_CONF = 0.35   # the reference model's own detections stand in for ground truth
RECALL_DROP_THRESHOLD = 0.10
VERDICT_CLASSES = ('person', 'car')   # the classes that matter most on these cameras
SHEET_COLOR_STOCK = 'orange'
SHEET_COLOR_CANDIDATE = 'lime'
MAX_SHEET_FRAMES = 8


# ------------------------------------------------------------------- matching

def iou_xyxy(a, b):
    """IoU of two (x1, y1, x2, y2) boxes. 0.0 for degenerate/non-overlapping boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_count(dets, refs, iou_thresh=IOU_THRESHOLD):
    """Greedy match, one class, one image.

    dets: [(box, conf), ...] (any order). refs: [box, ...]. Detections are
    tried highest-confidence first; each ref box can be claimed at most once,
    by whichever remaining detection overlaps it most (IoU >= iou_thresh).
    Returns the number of matched detections.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    used = [False] * len(refs)
    matched = 0
    for i in order:
        box, _conf = dets[i]
        best_iou, best_j = 0.0, -1
        for j, rbox in enumerate(refs):
            if used[j]:
                continue
            v = iou_xyxy(box, rbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            used[best_j] = True
            matched += 1
    return matched


def class_counts(dets_by_class, refs_by_class, classes, iou_thresh=IOU_THRESHOLD):
    """One image, one model: {class: dict(det=, ref=, matched=)} for `classes`."""
    out = {}
    for cls in classes:
        dets = dets_by_class.get(cls, [])
        refs = refs_by_class.get(cls, [])
        out[cls] = dict(det=len(dets), ref=len(refs), matched=match_count(dets, refs, iou_thresh))
    return out


def aggregate(per_frame_counts, classes):
    """Sum a list of class_counts() results (one per image) into totals."""
    totals = {cls: dict(det=0, ref=0, matched=0) for cls in classes}
    for counts in per_frame_counts:
        for cls in classes:
            for key in ('det', 'ref', 'matched'):
                totals[cls][key] += counts[cls][key]
    return totals


def recall_precision(totals):
    """{class: dict(det, ref, matched)} -> same plus recall/precision (None when undefined)."""
    out = {}
    for cls, t in totals.items():
        recall = t['matched'] / t['ref'] if t['ref'] else None
        precision = t['matched'] / t['det'] if t['det'] else None
        out[cls] = dict(t, recall=recall, precision=precision)
    return out


def verdict(stock_rp, candidate_rp, classes=VERDICT_CLASSES, drop_threshold=RECALL_DROP_THRESHOLD):
    """'better' / 'worse' / 'mixed' on the given classes (person+car by default).

    'worse' if the candidate's recall on any of `classes` is more than
    `drop_threshold` below stock's recall on that class - this overrides
    everything else, since a regression on the classes that matter most
    should never be masked by a gain elsewhere. Otherwise 'better' if the
    candidate is ahead (and never behind) on every class in `classes`, else
    'mixed'.
    """
    diffs = {}
    for cls in classes:
        stock_recall = stock_rp.get(cls, {}).get('recall') or 0.0
        candidate_recall = candidate_rp.get(cls, {}).get('recall') or 0.0
        diffs[cls] = candidate_recall - stock_recall
    if any(d < -drop_threshold for d in diffs.values()):
        return 'worse'
    if all(d >= 0 for d in diffs.values()) and any(d > 0 for d in diffs.values()):
        return 'better'
    return 'mixed'


# -------------------------------------------------------------- class mapping

def restrict_names(names, target_classes):
    """{index: name} from a model's .names, kept only where name (case-
    insensitively) matches one of target_classes, remapped to that exact
    spelling. A model that doesn't have a class is simply absent, same as
    the rest of this tool's "unmapped classes are dropped" convention.
    """
    lower_targets = {t.lower(): t for t in target_classes}
    out = {}
    for idx, name in names.items():
        key = str(name).strip().lower()
        if key in lower_targets:
            out[idx] = lower_targets[key]
    return out


def boxes_by_class(boxes_xyxy, boxes_cls, boxes_conf, idx_to_name):
    """Raw per-box (xyxy, class index, confidence) triples -> {class name: [(box, conf), ...]},
    restricted to idx_to_name (classes outside it are dropped)."""
    out = {}
    for box, cls, conf in zip(boxes_xyxy, boxes_cls, boxes_conf):
        name = idx_to_name.get(int(cls))
        if name is None:
            continue
        out.setdefault(name, []).append((tuple(box), float(conf)))
    return out


# ------------------------------------------------------------------- reporting

def format_table(stock_rp, candidate_rp, classes):
    def fmt(v):
        return f'{v:.3f}' if v is not None else 'n/a'

    header = (f"{'class':<10} {'stock det':>9} {'stock recall':>12} {'stock prec':>10}  "
              f"{'cand det':>8} {'cand recall':>11} {'cand prec':>9}")
    lines = [header]
    for cls in classes:
        s = stock_rp.get(cls, {})
        c = candidate_rp.get(cls, {})
        lines.append(
            f"{cls:<10} {s.get('det', 0):>9} {fmt(s.get('recall')):>12} {fmt(s.get('precision')):>10}  "
            f"{c.get('det', 0):>8} {fmt(c.get('recall')):>11} {fmt(c.get('precision')):>9}")
    return '\n'.join(lines)


def overall_line(stock_rp, candidate_rp, classes):
    def totals(rp):
        matched = sum(rp[c]['matched'] for c in classes)
        ref = sum(rp[c]['ref'] for c in classes)
        det = sum(rp[c]['det'] for c in classes)
        recall = matched / ref if ref else None
        precision = matched / det if det else None
        return recall, precision

    def fmt(v):
        return f'{v:.3f}' if v is not None else 'n/a'

    s_recall, s_prec = totals(stock_rp)
    c_recall, c_prec = totals(candidate_rp)
    return (f"{'overall':<10} stock recall={fmt(s_recall)} precision={fmt(s_prec)}   "
            f"candidate recall={fmt(c_recall)} precision={fmt(c_prec)}")


# ---------------------------------------------------------------- contact sheet

def draw_detections(draw, dets_by_class, color):
    for cls, dets in dets_by_class.items():
        for box, _conf in dets:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1 - 11)), cls, fill=color)


def build_contact_sheet(pairs, out_path, max_frames=MAX_SHEET_FRAMES):
    """pairs: [(image_path, stock_dets_by_class, candidate_dets_by_class), ...].
    Stacks up to max_frames rows, each row stock (left, orange) | candidate (right, lime)."""
    from PIL import Image, ImageDraw
    rows = []
    for img_path, stock_dets, candidate_dets in pairs[:max_frames]:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            left, right = im.copy(), im.copy()
        draw_detections(ImageDraw.Draw(left), stock_dets, SHEET_COLOR_STOCK)
        draw_detections(ImageDraw.Draw(right), candidate_dets, SHEET_COLOR_CANDIDATE)
        row = Image.new('RGB', (left.width + right.width, left.height))
        row.paste(left, (0, 0))
        row.paste(right, (left.width, 0))
        rows.append(row)
    if not rows:
        return
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    sheet = Image.new('RGB', (width, height), 'black')
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(out_path)


# ---------------------------------------------------------------------- runner

def run_model(model, img_path, conf, idx_to_name):
    """One image, device='mps' (this Mac OOMs on bigger batches). Returns {class: [(box, conf), ...]}."""
    result = model.predict(str(img_path), conf=conf, verbose=False, device='mps')[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {}
    return boxes_by_class(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), idx_to_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', default=str(Path.home() / '.clearcam-rf-build' / 'own' / 'selected' / 'holdout' / 'images'),
                         help='unlabelled holdout frames from collect_frames.py select')
    parser.add_argument('--candidate', required=True, help='candidate model weights (.pt)')
    parser.add_argument('--stock', default='yolo11s.pt', help='stock detector (default yolo11s.pt)')
    parser.add_argument('--reference', default='yolo11x.pt',
                         help=f'reference model whose detections at conf {REFERENCE_CONF} stand in for '
                              'ground truth (default yolo11x.pt)')
    parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold for stock/candidate (default 0.4)')
    parser.add_argument('--ignore', action='append', default=[],
                         help="'camera=x1,y1,x2,y2[:class,...]': reference boxes centred there are not real "
                              "(same areas as roboflow_dataset.py --ignore)")
    parser.add_argument('--sheet', default=None, help='optional contact sheet path (e.g. compare.jpg)')
    args = parser.parse_args()

    holdout_dir = Path(args.holdout).expanduser()
    images = tfc.list_images(holdout_dir)
    if not images:
        sys.exit(f'no images found under {holdout_dir}')

    from ultralytics import YOLO
    print(f'{len(images)} holdout frames; loading models...', flush=True)
    stock_model = YOLO(args.stock)
    candidate_model = YOLO(args.candidate)
    reference_model = YOLO(args.reference)

    stock_names = restrict_names({i: stock_model.names[i] for i in range(len(stock_model.names))}, TARGET_CLASSES)
    candidate_names = restrict_names({i: candidate_model.names[i] for i in range(len(candidate_model.names))}, TARGET_CLASSES)
    reference_names = restrict_names({i: reference_model.names[i] for i in range(len(reference_model.names))}, TARGET_CLASSES)

    # Reference first, for every frame, so boxes that never move (a post the
    # reference also calls a 'person' at night) can be dropped before scoring.
    reference_all = [run_model(reference_model, img_path, REFERENCE_CONF, reference_names) for img_path in images]
    flat = [(static_filter.camera_of(img_path.name),
             [(cls, box) for cls, dets in ref.items() for box, _conf in dets]) for img_path, ref in zip(images, reference_all)]
    masks = static_filter.static_mask(flat)
    ignore_areas = static_filter.parse_ignore(args.ignore)
    for (camera, boxes), mask in zip(flat, masks):
        for j, (cls, box) in enumerate(boxes):
            if static_filter.ignored(camera, cls, box, ignore_areas): mask[j] = False
    reference_filtered = []
    for (_camera, boxes), mask in zip(flat, masks):
        kept = {}
        for (cls, box), keep in zip(boxes, mask):
            if keep: kept.setdefault(cls, []).append(box)
        reference_filtered.append(kept)
    dropped = sum(m.count(False) for m in masks)
    print(f'reference: {dropped} stationary movable-class boxes dropped as scenery', flush=True)

    stock_frame_counts, candidate_frame_counts, sheet_pairs = [], [], []
    for n, img_path in enumerate(images):
        stock_dets = run_model(stock_model, img_path, args.conf, stock_names)
        candidate_dets = run_model(candidate_model, img_path, args.conf, candidate_names)
        reference_boxes = reference_filtered[n]

        stock_frame_counts.append(class_counts(stock_dets, reference_boxes, TARGET_CLASSES))
        candidate_frame_counts.append(class_counts(candidate_dets, reference_boxes, TARGET_CLASSES))
        if args.sheet and len(sheet_pairs) < MAX_SHEET_FRAMES:
            sheet_pairs.append((img_path, stock_dets, candidate_dets))
        if (n + 1) % 25 == 0:
            print(f'processed {n + 1}/{len(images)}', flush=True)

    stock_rp = recall_precision(aggregate(stock_frame_counts, TARGET_CLASSES))
    candidate_rp = recall_precision(aggregate(candidate_frame_counts, TARGET_CLASSES))

    print()
    print(format_table(stock_rp, candidate_rp, TARGET_CLASSES))
    print(overall_line(stock_rp, candidate_rp, TARGET_CLASSES))
    result = verdict(stock_rp, candidate_rp)
    print(f'\nverdict: candidate is {result} on {"+".join(VERDICT_CLASSES)}')

    if args.sheet:
        build_contact_sheet(sheet_pairs, args.sheet)
        print(f'contact sheet: {args.sheet}')


if __name__ == '__main__':
    main()
