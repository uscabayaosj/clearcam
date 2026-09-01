#!/bin/bash
# Run a training pass on this Mac's corrections and install the result for the app.
#   script/train.sh            # fine-tunes the Small detector (default in Settings)
#   script/train.sh n          # or the Nano one
#   script/train.sh s 30       # size and epochs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIZE="${1:-s}"; EPOCHS="${2:-20}"
DATA="${CLEARCAM_DATA_DIR:-$HOME/Library/Application Support/ClearCam/Data}"
[ -x "$ROOT/.venv-train/bin/python" ] || { echo "Run script/setup_training.sh once first." >&2; exit 1; }
# Work inside the data directory so downloaded checkpoints never land in the repo.
mkdir -p "$DATA/training" && cd "$DATA/training" && exec "$ROOT/.venv-train/bin/python" "$ROOT/script/train_from_corrections.py" --data "$DATA" --size "$SIZE" --epochs "$EPOCHS"
