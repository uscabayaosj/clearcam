"""Build-time export of the detection model to Core ML (run in an export venv).

Usage: <export-venv>/bin/python script/export_coreml.py [--out models/yolo11n.mlpackage]

Requires `ultralytics` and `coremltools` in the running interpreter. The
exported .mlpackage embeds an NMS pipeline, so the app-side wrapper
(detection/coreml_yolo.py) reads boxes and confidences directly.

Licensing note: Ultralytics YOLO11 weights and export code are AGPL-3.0.
Fine for personal builds; distribution requires an Ultralytics commercial
license or a swap to an Apache-licensed model.
"""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'models/yolo11n.mlpackage'))
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    exported = Path(model.export(format='coreml', nms=True, imgsz=640))
    target = Path(args.out)
    if target.exists(): shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(target))
    print(target)


if __name__ == '__main__':
    main()
