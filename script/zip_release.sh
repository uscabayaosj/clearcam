#!/bin/bash
# Cut a distributable zip of dist/ClearCam.app.
#
# The app must leave ~/Documents before it is verified or zipped: iCloud's
# file provider re-stamps com.apple.FinderInfo on anything under it, and a
# strict codesign verify (and therefore a stable identity on the installing
# Mac) rejects that "detritus". Stage in the system temp dir, strip, verify,
# zip without resource forks, then move only the zip back.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/ClearCam.app"
OUT="${1:-$ROOT/dist/ClearCam-m1.zip}"
[ -d "$APP" ] || { echo "no $APP; build first" >&2; exit 1; }
T="$(mktemp -d /tmp/clearcam-zip.XXXXXX)"
trap 'rm -rf "$T"' EXIT
ditto --norsrc "$APP" "$T/ClearCam.app"
xattr -cr "$T/ClearCam.app"
codesign --verify --deep --strict "$T/ClearCam.app"
(cd "$T" && ditto -c -k --norsrc --keepParent ClearCam.app ClearCam-m1.zip)
mv -f "$T/ClearCam-m1.zip" "$OUT"
echo "ZIP $(du -h "$OUT" | cut -f1) sha256:$(shasum -a 256 "$OUT" | cut -c1-16) $(codesign -dv "$T/ClearCam.app" 2>&1 | grep -o 'Authority=.*' | head -1)"
