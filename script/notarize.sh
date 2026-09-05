#!/bin/bash
# Notarize and staple dist/ClearCam.app, then cut the distributable zip.
#
# One-time setup (account holder, in a Terminal — the password is typed there,
# never stored in this repo):
#   xcrun notarytool store-credentials ClearCam --apple-id <Apple ID email> --team-id XU3LCVCLZC
# using an app-specific password from https://account.apple.com → Sign-In and
# Security → App-Specific Passwords. The profile name can be overridden with
# CLEARCAM_NOTARY_PROFILE. The app must have been packaged with a Developer ID
# Application certificate (package_macos.py picks it up automatically).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/ClearCam.app"
OUT="${1:-$ROOT/dist/ClearCam-m1.zip}"
PROFILE="${CLEARCAM_NOTARY_PROFILE:-ClearCam}"
[ -d "$APP" ] || { echo "no $APP; build first (bash script/build_and_run.sh --build-only)" >&2; exit 1; }
codesign -dv "$APP" 2>&1 | grep -q 'Authority=Developer ID Application' \
  || { echo "dist/ClearCam.app is not signed with a Developer ID certificate; notarization would be rejected." >&2; exit 1; }
T="$(mktemp -d /tmp/clearcam-notarize.XXXXXX)"
trap 'rm -rf "$T"' EXIT
ditto --norsrc "$APP" "$T/ClearCam.app"
xattr -cr "$T/ClearCam.app"
codesign --verify --deep --strict "$T/ClearCam.app"
(cd "$T" && ditto -c -k --norsrc --keepParent ClearCam.app upload.zip)
echo "Submitting $(du -h "$T/upload.zip" | cut -f1) to Apple's notary service (a large upload; expect several minutes)…"
xcrun notarytool submit "$T/upload.zip" --keychain-profile "$PROFILE" --wait --output-format json > "$T/result.json" || true
STATUS="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("status",""))' "$T/result.json" 2>/dev/null || true)"
ID="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("id",""))' "$T/result.json" 2>/dev/null || true)"
if [ "$STATUS" != "Accepted" ]; then
  echo "Notarization did not succeed (status: ${STATUS:-unknown}). Apple's log follows:" >&2
  [ -n "$ID" ] && xcrun notarytool log "$ID" --keychain-profile "$PROFILE" >&2 || cat "$T/result.json" >&2
  exit 1
fi
xcrun stapler staple "$T/ClearCam.app"
spctl -a -vv -t exec "$T/ClearCam.app"
# The stapled ticket lives in the bundle, so re-zip after stapling and keep the stapled copy.
(cd "$T" && ditto -c -k --norsrc --keepParent ClearCam.app ClearCam-m1.zip)
mv -f "$T/ClearCam-m1.zip" "$OUT"
rm -rf "$APP" && ditto --norsrc "$T/ClearCam.app" "$APP"
echo "NOTARIZED $(du -h "$OUT" | cut -f1) sha256:$(shasum -a 256 "$OUT" | cut -c1-16) submission:$ID"
