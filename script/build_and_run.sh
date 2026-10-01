#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-run}"
case "$MODE" in run|--verify|--build-only|--debug|--logs|--telemetry) ;; *) echo "Usage: $0 [--verify|--build-only|--debug|--logs|--telemetry]"; exit 2 ;; esac
if pgrep -x ClearCam >/dev/null; then
  pkill -TERM -x ClearCam
  for _ in {1..40}; do pgrep -x ClearCam >/dev/null || break; sleep 1; done
  if pgrep -x ClearCam >/dev/null; then echo "Quit ClearCam before rebuilding."; exit 1; fi
fi
cd "$TASK_ROOT"
# The MLX describer (fast, and the only way to run the 8B model) is bundled
# only when the build knows where mlx-vlm is installed. Default to the build
# venv so a plain rebuild can't silently fall back to the tinygrad describer.
DEFAULT_MLX_SITE="$HOME/.clearcam-mlx-venv/lib/python3.11/site-packages"
if [ -z "${CLEARCAM_MLX_SITE:-}" ] && [ -d "$DEFAULT_MLX_SITE/mlx_vlm" ]; then
  export CLEARCAM_MLX_SITE="$DEFAULT_MLX_SITE"
fi
# The bundled 2B describer is copied from the HF cache, which cache cleaners
# wipe; a copy kept outside ~/.cache survives that.
if [ -z "${CLEARCAM_MLX_MODELS:-}" ] && [ -f "$HOME/.clearcam-mlx-models/Qwen3-VL-2B-Instruct-4bit/config.json" ]; then
  export CLEARCAM_MLX_MODELS="$HOME/.clearcam-mlx-models"
fi
[ -n "${CLEARCAM_MLX_SITE:-}" ] || echo "warning: no CLEARCAM_MLX_SITE; descriptions will use the slower tinygrad 2B model only" >&2
swift build --package-path macos
TASK_BINARY="$(swift build --package-path macos --show-bin-path)/ClearCam"
.venv/bin/python script/package_macos.py --binary "$TASK_BINARY"
TASK_APP="$TASK_ROOT/dist/ClearCam.app"
case "$MODE" in
  --build-only) exit 0 ;;
  --debug) /usr/bin/lldb -- "$TASK_APP/Contents/MacOS/ClearCam" ;;
  *) /usr/bin/open -n "$TASK_APP" ;;
esac
case "$MODE" in
  --verify) sleep 2; pgrep -x ClearCam >/dev/null; echo "ClearCam process launched; engine readiness is checked separately." ;;
  --logs) /usr/bin/log stream --info --style compact --predicate 'process == "ClearCam"' ;;
  --telemetry) /usr/bin/log stream --info --style compact --predicate 'subsystem == "com.clearcam.mac.alpha"' ;;
esac
