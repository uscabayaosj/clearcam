"""Download a Roboflow-trained detector and convert it for the Neural Engine.

Training itself now happens on Roboflow's cloud; this script only downloads
the resulting weights and converts them to Core ML, mirroring what
train_from_corrections.py used to do locally.

Usage (training venv):
  <export-venv>/bin/python script/roboflow_model.py <version> \
      --data "$HOME/Library/Application Support/ClearCam/Data" \
      [--project P] [--workspace W] [--size s]
  <export-venv>/bin/python script/roboflow_model.py --rollback --data ... [--size s]

--size picks the detector slot the engine loads for that size (default 's');
it does not have to match whatever size Roboflow trained - the export step
converts to a fixed-input Core ML package regardless of the source
architecture's declared size.

Steps:
  1. Load the Roboflow config (workspace/project/api_key) from Data/roboflow.json.
  2. Download weights.pt via utils.roboflow_sync.download_weights into
     Data/training/roboflow-models/<project>-v<version>/.
  3. Load it with ultralytics YOLO(path); refuse anything that isn't a YOLO
     detection model (RF-DETR and friends aren't supported yet).
  4. Print the model's classes and how each maps onto ClearCam's canonical
     vocabulary (detection.coreml_yolo.build_canonical), showing dropped names.
  5. Export Core ML (nms=True, imgsz=640) and install it as
     Data/models/yolo11<size>-home.mlpackage, keeping the previous package as
     yolo11<size>-home.previous.mlpackage for --rollback.
  6. Write Data/models/yolo11<size>-home.json with provenance.
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import roboflow_sync
from detection.coreml_yolo import build_canonical

COCO = [l.strip() for l in (ROOT / 'models' / 'coco.names').read_text().splitlines() if l.strip()]

UNSUPPORTED_ARCH_MESSAGE = (
    "This Roboflow model can't be converted for the Neural Engine. Train it in Roboflow "
    "with a YOLO11 (or YOLOv8) object-detection architecture; RF-DETR and other "
    "architectures aren't supported yet."
)


def _models_dir(data_root):
    return Path(data_root) / 'models'


def _package_paths(data_root, size):
    models_dir = _models_dir(data_root)
    target = models_dir / f'yolo11{size}-home.mlpackage'
    previous = models_dir / f'yolo11{size}-home.previous.mlpackage'
    meta = models_dir / f'yolo11{size}-home.json'
    return target, previous, meta


def rollback(data_root, size):
    target, previous, meta = _package_paths(data_root, size)
    if previous.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(previous), str(target))
        print(f'rolled back: restored {target} from {previous.name}.')
    else:
        if target.exists():
            shutil.rmtree(target)
        if meta.exists():
            meta.unlink()
        print(f'no previous package to restore; removed {target.name} so the stock model is used.')
    print('Quit and reopen ClearCam to use it.')


def print_class_mapping(model_names):
    canonical, index_map = build_canonical(model_names, COCO)
    print(f'model reports {len(model_names)} classes:')
    dropped = []
    for i, name in enumerate(model_names):
        if i in index_map:
            print(f'  {name!r} -> {canonical[index_map[i]]!r} (canonical id {index_map[i]})')
        else:
            dropped.append(name)
    if dropped:
        print(f'dropped (no match in the ClearCam vocabulary): {dropped}')
    return canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('version', nargs='?', type=int, help='Roboflow project version to download')
    parser.add_argument('--data', default=str(Path.home() / 'Library/Application Support/ClearCam/Data'))
    parser.add_argument('--project', default=None)
    parser.add_argument('--workspace', default=None)
    parser.add_argument('--size', default='s', choices=['t', 's', 'm'],
                         help="the detector slot to install into (default: 's'; "
                              "does not need to match Roboflow's training size)")
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()

    data_root = Path(args.data).expanduser()
    if args.rollback:
        rollback(data_root, args.size)
        return

    if args.version is None:
        sys.exit('a Roboflow version is required unless --rollback is given')

    cfg = roboflow_sync.load_config(data_root)
    workspace = args.workspace or cfg.get('workspace', '')
    project = args.project or cfg.get('project', '')
    if not cfg.get('api_key'):
        sys.exit(f'Missing Roboflow api_key (set it via {roboflow_sync.CONFIG_FILE} under {data_root}).')
    if not workspace or not project:
        sys.exit('Missing workspace/project (pass --project/--workspace or configure them '
                  f'via {roboflow_sync.CONFIG_FILE} under {data_root}).')

    dest_dir = data_root / 'training' / 'roboflow-models' / f'{project}-v{args.version}'
    print(f'downloading weights for {workspace}/{project} v{args.version}...', flush=True)
    try:
        weights_path = roboflow_sync.download_weights(
            cfg, args.version, dest_dir, project=project, workspace=workspace)
    except roboflow_sync.RoboflowError as err:
        sys.exit(str(err))
    print(f'downloaded {weights_path}', flush=True)

    from ultralytics import YOLO
    try:
        model = YOLO(str(weights_path))
    except Exception as err:  # noqa: BLE001 - any load failure means "can't convert this"
        sys.exit(f'{UNSUPPORTED_ARCH_MESSAGE}\n(load error: {err})')
    if getattr(model, 'task', None) != 'detect':
        sys.exit(UNSUPPORTED_ARCH_MESSAGE)

    model_names = [model.names[i] for i in range(len(model.names))]
    print_class_mapping(model_names)

    print('exporting Core ML (Neural Engine)...', flush=True)
    exported = Path(model.export(format='coreml', nms=True, imgsz=640))

    target, previous, meta = _package_paths(data_root, args.size)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if previous.exists():
            shutil.rmtree(previous)
        shutil.move(str(target), str(previous))
        print(f'kept previous package as {previous.name} (use --rollback to restore it).')
    shutil.move(str(exported), str(target))

    meta.write_text(json.dumps(dict(
        source='roboflow', workspace=workspace, project=project, version=args.version,
        names=model_names, installed_at=datetime.datetime.now().isoformat(),
    ), indent=2))

    print(f'installed {target}')
    print('Installed. Quit and reopen ClearCam to use it. To roll back: bash script/roboflow_model.sh --rollback')


if __name__ == '__main__':
    main()
