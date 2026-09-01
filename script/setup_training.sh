#!/bin/bash
# One-time: a Python environment able to fine-tune and export the detector.
# Kept out of the app venv because torch is large and only training needs it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-train"
PY="$("$ROOT/.venv/bin/python" -c 'import sys;print(sys.executable)' 2>/dev/null || command -v python3.11 || command -v python3)"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# numpy 1.26 is deliberate: the Core ML converter misbehaves on numpy 2.
"$VENV/bin/pip" install -q "numpy==1.26.4" "torch==2.7.0" "torchvision==0.22.0" ultralytics coremltools
"$VENV/bin/python" -c "import ultralytics, torch, coremltools; print('training environment ready: ultralytics', ultralytics.__version__, 'torch', torch.__version__)"
