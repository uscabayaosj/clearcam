"""Apply counting-zone config to the running ClearCam engine.

  python script/set_count_zones.py zones.json

zones.json: {"<camera name>": [<zone>, ...], ...} where each <zone> is
{"id": str, "name": str, "polygon": [[x, y], ...] (0-1, 3-20 points),
"classes": [<class name>, ...], "metric": "passes" | "dwell"} -- the same
shape utils/count_zones.parse_zones validates, and the same shape the
per-camera settings modal in mainview.html saves through /edit_settings.

Talks to the engine the same way script/collect_frames.py does: finds the
running engine_bootstrap process, reads its listening port and session token
straight out of its own environment (never printed), and calls /edit_settings
over the local HTTP API. Replaces each named camera's whole count_zones list
(an empty list for a camera clears its zones); cameras not mentioned in
zones.json are left untouched.
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def engine_endpoint():
    """(base_url, token) of the running ClearCam engine, or None. Mirrors
    script/collect_frames.py's engine_endpoint() -- see there for why this
    reads the engine's own process environment instead of a config file."""
    pids = subprocess.run(['pgrep', '-f', 'engine_bootstrap'], capture_output=True, text=True).stdout.split()
    for pid in pids:
        listen = subprocess.run(['lsof', '-nP', '-a', '-p', pid, '-iTCP', '-sTCP:LISTEN'],
                                capture_output=True, text=True).stdout
        port = re.search(r'127\.0\.0\.1:(\d+) \(LISTEN\)', listen)
        env = subprocess.run(['ps', 'eww', pid], capture_output=True, text=True).stdout
        token = re.search(r'CLEARCAM_SESSION_TOKEN=(\S+)', env)
        if port and token:
            return f'http://127.0.0.1:{port.group(1)}', token.group(1)
    return None


def set_zones(base, token, camera, zones):
    """Raises urllib.error.HTTPError (with the engine's refusal message as
    the body) if `zones` fails validation server-side."""
    params = urllib.parse.urlencode({'cam': camera, 'count_zones': json.dumps(zones)})
    req = urllib.request.Request(f'{base}/edit_settings?{params}',
                                  headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help='JSON file: {"<camera name>": [<zone>, ...], ...}')
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if not isinstance(config, dict):
        sys.exit('config must be a JSON object of {"<camera name>": [<zone>, ...], ...}')

    endpoint = engine_endpoint()
    if endpoint is None:
        sys.exit('No running ClearCam engine found (engine_bootstrap is not running).')
    base, token = endpoint

    failures = 0
    for camera, zones in config.items():
        if not isinstance(zones, list):
            print(f'{camera}: skipped -- zones must be a list')
            failures += 1
            continue
        try:
            set_zones(base, token, camera, zones)
            print(f'{camera}: {len(zones)} zone(s) applied')
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace')
            print(f'{camera}: refused -- {body}')
            failures += 1
        except urllib.error.URLError as e:
            print(f'{camera}: could not reach the engine -- {e}')
            failures += 1

    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
