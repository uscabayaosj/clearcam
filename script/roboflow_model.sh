#!/bin/bash
# Download a Roboflow-trained detector and convert it for the Neural Engine.
#   bash script/roboflow_model.sh 3                       # install version 3 into the 's' slot
#   bash script/roboflow_model.sh 3 --size n              # install into the 'n' slot
#   bash script/roboflow_model.sh 3 --project P --workspace W
#   bash script/roboflow_model.sh --rollback               # undo the last install
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CLEARCAM_DATA_DIR:-$HOME/Library/Application Support/ClearCam/Data}"
[ -x "${CLEARCAM_TRAIN_VENV:-$HOME/.clearcam-train-venv}/bin/python" ] || { echo "Run script/setup_training.sh once first." >&2; exit 1; }
# Work inside the data directory so downloads never land in the repo.
mkdir -p "$DATA/training" && cd "$DATA/training" && exec "${CLEARCAM_TRAIN_VENV:-$HOME/.clearcam-train-venv}/bin/python" -u "$ROOT/script/roboflow_model.py" --data "$DATA" "$@"
