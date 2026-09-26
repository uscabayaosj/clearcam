"""Evaluate a fine-tuned checkpoint against the stock detector, per class, and
install it for ClearCam only if it does not lose ground on the classes the
stock model already knows.

Usage (training venv):
  ~/.clearcam-train-venv/bin/python script/evaluate_and_install.py \
      --weights ".../yolo11s-home/run/weights/best.pt" \
      --data ".../yolo11s-home/dataset/home.yaml" [--install]

Why a separate step: a long training run can be killed (memory, sleep) after
its best checkpoint is already on disk. This salvages that checkpoint with a
memory-light validation (small batch) and a per-class comparison, instead of
the averaged score that hides which classes moved.
"""
import argparse
import shutil
from pathlib import Path

OWNER_CLASSES = ['person', 'car', 'truck', 'bicycle', 'dog', 'stroller', 'child', 'scooter']
MAX_RELATIVE_DROP = 0.10   # on classes the stock model already knows


def per_class(metrics, names):
    """{class name: (mAP50, mAP50-95)} for classes present in the validation set."""
    box = metrics.box
    out = {}
    for position, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(position)
        # The stock model has 80 names; the validation labels also use the new
        # classes (80+), which it cannot know about.
        name = names.get(int(cls)) if isinstance(names, dict) else (names[int(cls)] if int(cls) < len(names) else None)
        if name is not None:
            out[name] = (float(ap50), float(ap))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--size', default='s', choices=['n', 's'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--models-dir', default=str(Path.home() / 'Library/Application Support/ClearCam/Data/models'))
    args = parser.parse_args()
    from ultralytics import YOLO

    common = dict(data=args.data, imgsz=640, batch=args.batch, device='mps', workers=2, plots=False, verbose=False)
    tuned = YOLO(args.weights)
    print('validating tuned model…', flush=True)
    tuned_scores = per_class(tuned.val(**common), tuned.names)
    stock = YOLO(f'yolo11{args.size}.pt')
    print('validating stock model…', flush=True)
    stock_scores = per_class(stock.val(**common), stock.names)

    print(f"\n{'class':<10} {'stock mAP50':>12} {'tuned mAP50':>12}")
    for name in OWNER_CLASSES:
        s = stock_scores.get(name, (None, None))[0]
        t = tuned_scores.get(name, (None, None))[0]
        fmt = lambda v: f'{v:.3f}' if v is not None else 'n/a'
        print(f'{name:<10} {fmt(s):>12} {fmt(t):>12}')

    # Gate on classes both models know and the validation set contains.
    shared = [n for n in stock_scores if n in tuned_scores]
    stock_mean = sum(stock_scores[n][1] for n in shared) / max(len(shared), 1)
    tuned_mean = sum(tuned_scores[n][1] for n in shared) / max(len(shared), 1)
    drop = (stock_mean - tuned_mean) / stock_mean if stock_mean else 0.0
    print(f'\nshared classes ({len(shared)}): stock mAP50-95 {stock_mean:.3f}, tuned {tuned_mean:.3f}, change {-drop:+.1%}')
    passed = drop <= MAX_RELATIVE_DROP
    print('quality gate:', 'PASS' if passed else f'FAIL (drop over {MAX_RELATIVE_DROP:.0%})')

    if not args.install:
        return
    if not passed:
        print('not installing.')
        return
    exported = Path(tuned.export(format='coreml', nms=True, imgsz=640))
    target = Path(args.models_dir) / f'yolo11{args.size}-home.mlpackage'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(exported), str(target))
    print(f'installed {target}; quit and reopen ClearCam to use it.')


if __name__ == '__main__':
    main()
