#!/bin/bash
# Build a training set from public Roboflow Universe datasets and upload it
# into your own Roboflow project, so training happens in Roboflow's cloud.
#   bash script/roboflow_dataset.sh --source marcu/stroller-vxfbx --source kpz3/stroller-tdpar
#   bash script/roboflow_dataset.sh --source kicksquad/scooter-detect --dry-run
#   bash script/roboflow_dataset.sh --source ws/proj:3 --target-project clearcam-home --max-images 800
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CLEARCAM_DATA_DIR:-$HOME/Library/Application Support/ClearCam/Data}"
[ -x "${CLEARCAM_TRAIN_VENV:-$HOME/.clearcam-train-venv}/bin/python" ] || { echo "Run script/setup_training.sh once first." >&2; exit 1; }
# Work inside the data directory so downloads never land in the repo.
mkdir -p "$DATA/training" && cd "$DATA/training" && exec "${CLEARCAM_TRAIN_VENV:-$HOME/.clearcam-train-venv}/bin/python" -u "$ROOT/script/roboflow_dataset.py" --data "$DATA" "$@"
