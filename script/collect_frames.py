"""Collect stills from ClearCam's own cameras for training, then keep a varied subset.

  python script/collect_frames.py collect --hours 12 --interval 120
  python script/collect_frames.py select --per-camera 150

`collect` asks the running ClearCam engine for one frame per camera (the
/live_view stream, without boxes), so it needs no camera credentials: the
engine already has the frames in memory. Frames go to <work>/raw/<camera>/.

`select` keeps up to --per-camera frames per camera that differ visibly from
each other, spread across the hours they were taken, and splits them into
train/valid (uploaded to Roboflow) and holdout (never uploaded; used to judge
a trained model on this home's own cameras). The raw frames are deleted
afterwards unless --keep-raw is given.

Everything stays under <work> (default ~/.clearcam-rf-build/own), outside
iCloud-synced folders.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_WORK = Path.home() / '.clearcam-rf-build' / 'own'


# ------------------------------------------------------------------ engine

def engine_endpoint():
    """(base_url, token) of the running ClearCam engine, or None."""
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


def _get(base, token, path, timeout=10):
    req = urllib.request.Request(base + path, headers={'Authorization': f'Bearer {token}'})
    return urllib.request.urlopen(req, timeout=timeout)


def list_cameras(base, token):
    with _get(base, token, '/list_cameras') as resp:
        return list(json.loads(resp.read()).keys())


def first_jpeg(stream, limit=8 * 1024 * 1024):
    """Read the first JPEG part out of a multipart/x-mixed-replace stream."""
    header = b''
    while b'\r\n\r\n' not in header:
        chunk = stream.read(1)
        if not chunk: return None
        header += chunk
        if len(header) > 4096: return None
    length = re.search(rb'Content-Length:\s*(\d+)', header)
    if not length: return None
    size = int(length.group(1))
    if size <= 0 or size > limit: return None
    data = b''
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk: return None
        data += chunk
    return data if data[:2] == b'\xff\xd8' else None


def grab(base, token, camera):
    path = '/live_view?' + urllib.parse.urlencode({'cam': camera, 'boxes': '0'})
    with _get(base, token, path, timeout=20) as resp:
        return first_jpeg(resp)


def collect(work, hours, interval):
    raw = Path(work) / 'raw'
    deadline = time.time() + hours * 3600
    saved = 0
    print(f'collecting every {interval}s for {hours}h into {raw}', flush=True)
    while time.time() < deadline:
        started = time.time()
        endpoint = engine_endpoint()   # re-read each round: the app may have been relaunched
        if endpoint is None:
            print(f'{datetime.now():%H:%M} ClearCam is not running; waiting', flush=True)
        else:
            base, token = endpoint
            try:
                cameras = list_cameras(base, token)
            except Exception as err:
                print(f'{datetime.now():%H:%M} could not list cameras: {err}', flush=True)
                cameras = []
            for camera in cameras:
                try:
                    jpg = grab(base, token, camera)
                except Exception as err:
                    print(f'{datetime.now():%H:%M} {camera}: {err}', flush=True)
                    continue
                if not jpg: continue
                folder = raw / camera
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f'{datetime.now():%Y%m%d-%H%M%S}.jpg').write_bytes(jpg)
                saved += 1
            if saved and saved % 40 < len(cameras):
                print(f'{datetime.now():%H:%M} {saved} frames saved', flush=True)
        time.sleep(max(0, interval - (time.time() - started)))
    print(f'done: {saved} frames saved', flush=True)


# ------------------------------------------------------------------ select

def thumbnail(path, size=(32, 18)):
    from PIL import Image
    with Image.open(path) as im:
        return list(im.convert('L').resize(size).tobytes())


def difference(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def pick_varied(frames, limit, min_diff=6.0, thumb=thumbnail):
    """Greedy pick of frames that differ from everything already picked.

    frames: time-ordered paths. Frames are visited round-robin across hours
    so a busy hour can't crowd out the night.
    """
    by_hour = {}
    for f in frames:
        by_hour.setdefault(f.name[:11], []).append(f)   # YYYYMMDD-HH
    order = []
    queues = [list(v) for _, v in sorted(by_hour.items())]
    while any(queues):
        for q in queues:
            if q: order.append(q.pop(0))
    picked, thumbs = [], []
    for f in order:
        t = thumb(f)
        if all(difference(t, other) >= min_diff for other in thumbs):
            picked.append(f)
            thumbs.append(t)
            if len(picked) >= limit: break
    return sorted(picked)


def split_by_hour(name, holdout_hours=(1, 5, 13, 17), valid_hours=(3, 9, 15, 21)):
    """Split by the hour a frame was taken (name 'YYYYMMDD-HHMMSS.jpg').

    Fixed cameras make neighbouring frames near-identical, so a per-frame
    split would leak almost the same picture into train and holdout. Whole
    hours held back keep the holdout honest.
    """
    hour = int(name[9:11])
    return 'holdout' if hour in holdout_hours else ('valid' if hour in valid_hours else 'train')


def split_for(name, camera):
    """Deterministic split by hash: ~70% train, ~15% valid, ~15% holdout."""
    h = int(hashlib.sha1(f'{camera}/{name}'.encode()).hexdigest(), 16) % 100
    return 'train' if h < 70 else ('valid' if h < 85 else 'holdout')


def select(work, per_camera, keep_raw=False, min_diff=6.0, stride=None):
    work = Path(work)
    raw = work / 'raw'
    out = work / 'selected'
    if out.exists(): shutil.rmtree(out)
    totals = {}
    for cam_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        frames = sorted(cam_dir.glob('*.jpg'))
        # --stride: every Nth frame, split by hour. Fixed cameras barely change
        # between frames (a passer-by moves the whole-frame difference very
        # little), so the difference-based pick keeps almost nothing.
        picked = frames[::stride][:per_camera] if stride else pick_varied(frames, per_camera, min_diff)
        slug = re.sub(r'[^a-z0-9]+', '-', cam_dir.name.lower()).strip('-')
        for f in picked:
            split = split_by_hour(f.name) if stride else split_for(f.name, cam_dir.name)
            dest = out / split / 'images'
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f'own-{slug}-{f.name}')
            totals.setdefault(cam_dir.name, {}).setdefault(split, 0)
            totals[cam_dir.name][split] += 1
        print(f'{cam_dir.name}: {len(frames)} collected, {len(picked)} kept {totals.get(cam_dir.name, {})}')
    if not keep_raw:
        shutil.rmtree(raw)
        print('raw frames deleted')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--hours', type=float, default=12)
    c.add_argument('--interval', type=int, default=120)
    s = sub.add_parser('select')
    s.add_argument('--per-camera', type=int, default=150)
    s.add_argument('--min-diff', type=float, default=6.0)
    s.add_argument('--keep-raw', action='store_true')
    s.add_argument('--stride', type=int, default=None,
                   help='keep every Nth frame and split by hour (suits fixed cameras)')
    for p in (c, s):
        p.add_argument('--work', default=str(DEFAULT_WORK))
    args = parser.parse_args()
    if args.cmd == 'collect':
        collect(args.work, args.hours, args.interval)
    else:
        select(args.work, args.per_camera, args.keep_raw, args.min_diff, args.stride)


if __name__ == '__main__':
    main()
