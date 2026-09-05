#!/bin/bash
# Finish Developer ID setup once Apple has issued the certificate.
#
# 1. Upload ~/Library/Application Support/ClearCam/signing/DeveloperID.certSigningRequest
#    at https://developer.apple.com/account/resources/certificates/add
#    (type: "Developer ID Application", profile type: G2 Sub-CA) and download the .cer.
# 2. bash script/install_developer_id.sh ~/Downloads/developerID_application.cer
# The private key was generated alongside the request and already lives in the
# login keychain, so importing the certificate completes the signing identity.
set -euo pipefail
CER="${1:-$HOME/Downloads/developerID_application.cer}"
[ -f "$CER" ] || { echo "certificate not found: $CER" >&2; exit 1; }
security import "$CER" -k "$HOME/Library/Keychains/login.keychain-db" 2>&1 | grep -v "already exists" || true
# Apple's intermediate; harmless if already present.
curl -fsSL https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer -o /tmp/DeveloperIDG2CA.cer \
  && security import /tmp/DeveloperIDG2CA.cer -k "$HOME/Library/Keychains/login.keychain-db" 2>/dev/null || true
if security find-identity -v -p codesigning | grep -q "Developer ID Application"; then
  security find-identity -v -p codesigning | grep "Developer ID Application"
  echo "Developer ID ready. Next: bash script/build_and_run.sh --build-only && bash script/notarize.sh"
else
  echo "The certificate imported but no signing identity formed; the private key may not match the request." >&2; exit 1
fi
