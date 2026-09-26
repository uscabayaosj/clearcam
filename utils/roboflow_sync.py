"""Push owner corrections to Roboflow so a wider dataset can be built from them.

Corrections already carry ground truth (utils.corrections): 'confirm' keeps
the boxes as-is, 'wrong_label' relabels the triggering box, 'not_object'
drops it. This module turns each correction into a Roboflow upload (an image
plus a Pascal VOC annotation) using nothing but the standard library, tracks
what has already been uploaded in a small ledger so re-running is safe, and
can later pull a merged dataset (local + Roboflow) back down for training.

REST endpoints used (Roboflow's classic dataset API, no SDK):
  POST https://api.roboflow.com/dataset/{project}/upload?api_key=...&name=...&split=train&batch=...
  POST https://api.roboflow.com/dataset/{project}/annotate/{id}?api_key=...&name=...
  GET  https://api.roboflow.com/{workspace}/{project}/{version}/yolov11?api_key=...
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from utils.corrections import load_corrections

CONFIG_FILE = 'roboflow.json'
_CONFIG_DEFAULTS = dict(enabled=False, workspace='', project='', api_key='')
_KNOWN_KEYS = tuple(_CONFIG_DEFAULTS.keys())

_sync_lock = threading.Lock()
_sync_thread = None


class RoboflowError(Exception):
    pass


# --------------------------------------------------------------------------- config

def _config_path(data_root):
    return Path(data_root) / CONFIG_FILE


def load_config(data_root):
    cfg = dict(_CONFIG_DEFAULTS)
    path = _config_path(data_root)
    if path.is_file():
        try:
            on_disk = json.loads(path.read_text())
        except ValueError:
            on_disk = {}
        for key in _KNOWN_KEYS:
            if key in on_disk:
                cfg[key] = on_disk[key]
    return cfg


def save_config(data_root, updates):
    cfg = load_config(data_root)
    for key in _KNOWN_KEYS:
        if key in updates:
            cfg[key] = updates[key]
    path = _config_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(cfg))
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)
    return cfg


def public_config(cfg):
    out = dict(cfg)
    key = out.pop('api_key', '') or ''
    out['has_key'] = bool(key)
    out['key_hint'] = key[-4:] if key else ''
    return out


# ------------------------------------------------------------------------ annotation

def annotation_xml(entry, class_names):
    width = entry.get('width') or 0
    height = entry.get('height') or 0
    verdict = entry.get('verdict')
    boxes = []
    for det in entry.get('detections', []):
        label = det.get('label', '')
        trigger = bool(det.get('trigger'))
        if verdict == 'not_object' and trigger:
            continue
        if verdict == 'wrong_label' and trigger:
            label = entry.get('label') or label
        boxes.append((label, det.get('box', [0, 0, 0, 0])))
    filename = entry.get('image', '')
    objects = ''
    for label, box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        objects += (
            '  <object>\n'
            f'    <name>{escape(str(label))}</name>\n'
            '    <bndbox>\n'
            f'      <xmin>{x1}</xmin>\n'
            f'      <ymin>{y1}</ymin>\n'
            f'      <xmax>{x2}</xmax>\n'
            f'      <ymax>{y2}</ymax>\n'
            '    </bndbox>\n'
            '  </object>\n'
        )
    return (
        '<annotation>\n'
        f'  <filename>{escape(str(filename))}</filename>\n'
        '  <size>\n'
        f'    <width>{int(width)}</width>\n'
        f'    <height>{int(height)}</height>\n'
        '    <depth>3</depth>\n'
        '  </size>\n'
        f'{objects}'
        '</annotation>\n'
    )


# --------------------------------------------------------------------------- ledger

def _roboflow_dir(data_root):
    return Path(data_root) / 'roboflow'


def _ledger_path(data_root):
    return _roboflow_dir(data_root) / 'uploaded.jsonl'


def _state_path(data_root):
    return _roboflow_dir(data_root) / 'state.json'


def uploaded_images(data_root):
    path = _ledger_path(data_root)
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        image = row.get('image')
        if image:
            out.add(image)
    return out


def _append_ledger(data_root, image, roboflow_id):
    path = _ledger_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(image=image, id=roboflow_id, time=time.time())) + '\n')


def _load_state(data_root):
    path = _state_path(data_root)
    if not path.is_file():
        return dict(uploaded_total=0, last_sync=None, last_error=None, last_error_time=None, syncing=False)
    try:
        state = json.loads(path.read_text())
    except ValueError:
        state = {}
    state.setdefault('uploaded_total', 0)
    state.setdefault('last_sync', None)
    state.setdefault('last_error', None)
    state.setdefault('last_error_time', None)
    state.setdefault('syncing', False)
    return state


def _save_state(data_root, state):
    path = _state_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state))
    temporary.replace(path)


def pending_entries(data_root):
    already = uploaded_images(data_root)
    store = Path(data_root) / 'corrections' / 'images'
    out = []
    for entry in load_corrections(data_root):
        image = entry.get('image')
        if not image or image in already:
            continue
        if not (store / image).is_file():
            continue
        out.append(entry)
    return out


def status(data_root):
    state = _load_state(data_root)
    corrections = load_corrections(data_root)
    state['total_corrections'] = len(corrections)
    state['pending'] = len(pending_entries(data_root))
    return state


# ------------------------------------------------------------------------- upload

def _default_opener(request, timeout=30):
    return urllib.request.urlopen(request, timeout=timeout)


def _read_json_response(resp):
    body = resp.read()
    try:
        return json.loads(body.decode('utf-8'))
    except ValueError:
        return {}


def _raise_http_error(err, api_key):
    try:
        body = err.read().decode('utf-8', 'replace')
    except Exception:
        body = ''
    if api_key:
        body = body.replace(api_key, '***')
    snippet = body[:300]
    raise RoboflowError(f'HTTP {err.code}: {snippet}')


def upload_image_and_annotation(cfg, image_path, xml, split='train', batch='ClearCam corrections', opener=None,
                                name=None):
    """Upload one image plus a prebuilt Pascal VOC annotation string.

    Shared by upload_entry (corrections, always split='train') and any other
    caller that needs an arbitrary split/batch name - e.g. building a
    training set from Roboflow Universe sources, which uploads to both
    'train' and 'valid' under a per-source batch name.
    """
    opener = opener or _default_opener
    api_key = cfg.get('api_key', '')
    project = cfg.get('project', '')
    image_path = Path(image_path)
    boundary = uuid.uuid4().hex
    name = name or image_path.name   # name as stored in Roboflow; must be unique in the project
    data = image_path.read_bytes()
    # Same shape as Roboflow's own SDK: name and split travel as form fields
    # beside the file part.
    fields = ''.join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'
        for field, value in (('name', name), ('split', split))
    )
    body = (
        fields +
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + data + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    from urllib.parse import quote
    upload_url = (
        f'https://api.roboflow.com/dataset/{project}/upload'
        f'?api_key={quote(api_key)}&name={quote(name)}&split={quote(split)}&batch={quote(batch)}'
    )
    request = urllib.request.Request(
        upload_url, data=body, method='POST',
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    )
    try:
        resp = opener(request, timeout=30)
    except urllib.error.HTTPError as err:
        _raise_http_error(err, api_key)
        return  # unreachable
    payload = _read_json_response(resp)
    if not (payload.get('success') or payload.get('duplicate')):
        raise RoboflowError(f'upload failed: {json.dumps(payload)[:300]}')
    image_id = payload.get('id')
    if not image_id:
        raise RoboflowError(f'upload response missing id: {json.dumps(payload)[:300]}')

    # Annotation goes as JSON {annotationFile, labelmap}, as the SDK sends it;
    # overwrite=true keeps a re-sync from tripping over an earlier attempt.
    annotate_url = (
        f'https://api.roboflow.com/dataset/{project}/annotate/{quote(str(image_id))}'
        f'?api_key={quote(api_key)}&name={quote(name)}.xml&overwrite=true'
    )
    request = urllib.request.Request(
        annotate_url, data=json.dumps({'annotationFile': xml, 'labelmap': None}).encode('utf-8'), method='POST',
        headers={'Content-Type': 'application/json'},
    )
    try:
        resp = opener(request, timeout=30)
    except urllib.error.HTTPError as err:
        if err.code == 409:
            return dict(id=image_id, annotated=True, already=True)   # already annotated in Roboflow
        _raise_http_error(err, api_key)
        return  # unreachable
    payload = _read_json_response(resp)
    if not payload.get('success'):
        raise RoboflowError(f'annotate failed: {json.dumps(payload)[:300]}')
    return dict(id=image_id, annotated=True)


def upload_entry(cfg, image_path, entry, class_names, opener=None):
    xml = annotation_xml(entry, class_names)
    return upload_image_and_annotation(cfg, image_path, xml, split='train', batch='ClearCam corrections', opener=opener)


def voc_xml_from_yolo_lines(lines, class_names, filename, width, height):
    """Pascal VOC XML for a set of YOLO-normalised label lines, using class NAMES.

    Mirrors annotation_xml's output shape (same tags Roboflow's classic
    upload/annotate API expects) but starts from plain 'cls cx cy w h' lines
    and pixel width/height, rather than a corrections-store entry.
    """
    objects = ''
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if cls < 0 or cls >= len(class_names):
            continue
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
        name = class_names[cls]
        objects += (
            '  <object>\n'
            f'    <name>{escape(str(name))}</name>\n'
            '    <bndbox>\n'
            f'      <xmin>{int(round(x1))}</xmin>\n'
            f'      <ymin>{int(round(y1))}</ymin>\n'
            f'      <xmax>{int(round(x2))}</xmax>\n'
            f'      <ymax>{int(round(y2))}</ymax>\n'
            '    </bndbox>\n'
            '  </object>\n'
        )
    return (
        '<annotation>\n'
        f'  <filename>{escape(str(filename))}</filename>\n'
        '  <size>\n'
        f'    <width>{int(width)}</width>\n'
        f'    <height>{int(height)}</height>\n'
        '    <depth>3</depth>\n'
        '  </size>\n'
        f'{objects}'
        '</annotation>\n'
    )


def sync(data_root, class_names, limit=100, opener=None):
    data_root = Path(data_root)
    if not _sync_lock.acquire(blocking=False):
        return status(data_root)
    try:
        cfg = load_config(data_root)
        state = _load_state(data_root)
        if not cfg.get('enabled'):
            state['last_error'] = 'Roboflow sync is not enabled.'
            state['last_error_time'] = time.time()
            state['syncing'] = False
            _save_state(data_root, state)
            return status(data_root)
        missing = [k for k in ('api_key', 'project') if not cfg.get(k)]
        if missing:
            state['last_error'] = f'Missing Roboflow config: {", ".join(missing)}.'
            state['last_error_time'] = time.time()
            state['syncing'] = False
            _save_state(data_root, state)
            return status(data_root)

        state['syncing'] = True
        state['last_error'] = None
        _save_state(data_root, state)

        images_dir = data_root / 'corrections' / 'images'
        entries = pending_entries(data_root)[:limit]
        uploaded = 0
        for entry in entries:
            image_path = images_dir / entry['image']
            try:
                result = upload_entry(cfg, image_path, entry, class_names, opener=opener)
            except RoboflowError as err:
                state['last_error'] = str(err)
                state['last_error_time'] = time.time()
                break
            _append_ledger(data_root, entry['image'], result['id'])
            uploaded += 1
        state['uploaded_total'] = state.get('uploaded_total', 0) + uploaded
        state['last_sync'] = time.time()
        state['syncing'] = False
        _save_state(data_root, state)
        return status(data_root)
    finally:
        _sync_lock.release()


def sync_in_background(data_root, class_names):
    global _sync_thread
    if _sync_thread is not None and _sync_thread.is_alive():
        return
    _sync_thread = threading.Thread(target=sync, args=(data_root, class_names), daemon=True)
    _sync_thread.start()


# ----------------------------------------------------------------------- download

def download_weights(cfg, version, dest_dir, project=None, workspace=None, opener=None):
    """Download the PyTorch weights for a trained Roboflow model version.

    Mirrors roboflow-python's Model.download: a first request to the
    'ptFile' endpoint returns {"weightsUrl": "<signed url>"} (the signed URL
    itself needs no api_key and must not be logged in full); a second, plain
    GET on that URL streams the actual weights file. Only the 'pt' format
    exists for Roboflow-trained YOLO models.

    Streams to dest_dir/'weights.pt' via a '.partial' file, then renames, so
    a failed/aborted download never leaves a corrupt file at the final path.
    Returns the Path to the written weights file.
    """
    opener = opener or _default_opener
    api_key = cfg.get('api_key', '')
    workspace = workspace or cfg.get('workspace', '')
    project = project or cfg.get('project', '')
    from urllib.parse import quote
    meta_url = f'https://api.roboflow.com/{workspace}/{project}/{version}/ptFile?api_key={quote(api_key)}'
    request = urllib.request.Request(meta_url, method='GET')
    try:
        resp = opener(request, timeout=60)
    except urllib.error.HTTPError as err:
        try:
            body = err.read().decode('utf-8', 'replace')
        except Exception:
            body = ''
        if api_key:
            body = body.replace(api_key, '***')
        message = body
        try:
            message = json.loads(body).get('message', body)
        except ValueError:
            pass
        snippet = str(message)[:300]
        raise RoboflowError(
            f'Roboflow refused the weights download: {snippet}. '
            'Weight downloads need a plan or academic access that includes them.'
        )
    payload = _read_json_response(resp)
    weights_url = payload.get('weightsUrl')
    if not weights_url:
        raise RoboflowError('Roboflow response did not include a weightsUrl (no message given).')

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / 'weights.pt'
    partial_path = dest_dir / 'weights.pt.partial'
    # weights_url is a pre-signed URL; never log it in full.
    request = urllib.request.Request(weights_url, method='GET')
    try:
        resp = opener(request, timeout=60)
    except urllib.error.HTTPError as err:
        try:
            body = err.read().decode('utf-8', 'replace')
        except Exception:
            body = ''
        raise RoboflowError(f'HTTP {err.code} downloading weights: {body[:300]}')
    try:
        with partial_path.open('wb') as stream:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    partial_path.replace(final_path)
    return final_path


def download_dataset(cfg, version, dest_dir, opener=None):
    opener = opener or _default_opener
    api_key = cfg.get('api_key', '')
    workspace = cfg.get('workspace', '')
    project = cfg.get('project', '')
    from urllib.parse import quote
    url = f'https://api.roboflow.com/{workspace}/{project}/{version}/yolov11?api_key={quote(api_key)}'
    # A first request may only start the export; poll until Roboflow reports it ready.
    deadline = time.time() + 600
    while True:
        request = urllib.request.Request(url, method='GET')
        try:
            resp = opener(request, timeout=30)
        except urllib.error.HTTPError as err:
            _raise_http_error(err, api_key)
            return  # unreachable
        payload = _read_json_response(resp)
        export = payload.get('export') or {}
        link = export.get('link') or payload.get('link')
        if link and payload.get('ready', True) is not False:
            break
        if time.time() > deadline:
            raise RoboflowError('Roboflow export did not become ready within 10 minutes')
        time.sleep(5)

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / 'dataset.zip'
    request = urllib.request.Request(link, method='GET')
    resp = opener(request, timeout=60)
    zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    yaml_candidates = list(dest_dir.rglob('data.yaml'))
    if not yaml_candidates:
        raise RoboflowError('data.yaml not found in downloaded dataset')
    return yaml_candidates[0]


# ------------------------------------------------------------------- class/label ops

def normalise_slug(value, field='workspace'):
    """Accept what people actually paste: a slug, a display name, or a project URL.

    Roboflow identifies workspaces and projects by URL slugs (lowercase,
    digits, hyphens). 'https://app.roboflow.com/my-ws/clearcam-home/1' gives
    my-ws for the workspace and clearcam-home for the project; 'My Home' becomes
    my-home. Empty input clears the field.
    """
    import re
    text = (value or '').strip()
    if not text: return ''
    match = re.search(r'roboflow\.com/([^/?#]+)(?:/([^/?#]+))?', text)
    if match:
        text = match.group(1) if field == 'workspace' else (match.group(2) or match.group(1))
    slug = re.sub(r'[^a-z0-9-]+', '-', text.lower()).strip('-')
    slug = re.sub(r'-{2,}', '-', slug)
    if not slug or len(slug) > 100:
        raise ValueError(f'{field} must be the Roboflow URL name (letters, numbers and dashes), at most 100 characters')
    return slug


def merge_class_names(base, extra):
    out = list(base)
    existing = set(base)
    for name in extra:
        if name not in existing:
            out.append(name)
            existing.add(name)
    return out


# Common synonyms seen in external/Universe datasets, mapped onto the
# ClearCam vocabulary before merging. Keys are looked up after the name has
# been lowercased, stripped, had a leading 'N-' numeric prefix removed, and
# had underscores turned into spaces - so 'fire_hydrant' needs no entry here
# (it becomes 'fire hydrant', matching COCO directly), but 'person_cane'
# needs one keyed as 'person cane'. Several source names may collapse onto
# the same target.
ALIASES = {
    'human': 'person',
    'old': 'person',
    'adolescent': 'person',
    'adult': 'person',
    'person cane': 'person',
    'person crutches': 'person',
    'person walking frame': 'person',
    'person wheelchair': 'person',
    'pedestrian': 'person',
    'people': 'person',
    'persons': 'person',
    'kids': 'child',
    'kid': 'child',
    'baby': 'child',
    'children': 'child',
    'ambulance': 'truck',
    'bike': 'bicycle',
    'bicycles': 'bicycle',
    'motorbike': 'motorcycle',
    'e-scooter': 'scooter',
    'kick scooter': 'scooter',
    'kickscooter': 'scooter',
    'pram': 'stroller',
    'prams': 'stroller',
    'baby carriage': 'stroller',
    'carriage': 'stroller',
    'buggy': 'stroller',
    'kickboard': 'scooter',
    'electric-kickboard': 'scooter',
    'kick-scooter': 'scooter',
    'electric scooter': 'scooter',
    'stop': 'stop sign',
    'cats': 'cat',
    'dogs': 'dog',
    'cars': 'car',
}


def normalise_class(name, base_names):
    """Normalise an external class name onto the ClearCam vocabulary.

    Lowercases and strips the name, drops a leading 'N-' numeric prefix
    (Universe datasets sometimes number their classes, e.g. '3-stroller'),
    turns underscores into spaces, applies ALIASES, then - if the result
    matches an existing name in base_names case-insensitively - returns that
    base_names entry (so case and any earlier-established spelling win).
    Otherwise returns the normalised (lowercased) name unchanged; callers
    that only keep a fixed vocabulary (COCO + configured new classes) treat
    a name that still doesn't match anything in base_names as unknown.
    """
    import re
    norm = (name or '').strip().lower()
    norm = re.sub(r'^\d+-', '', norm)
    norm = norm.replace('_', ' ').strip()
    norm = ALIASES.get(norm, norm)
    for existing in base_names:
        if existing.lower() == norm:
            return existing
    return norm


def build_index_map(src_names, merged_names):
    """Map each index of src_names to its index in merged_names.

    Uses normalise_class to resolve aliases/case before matching. A source
    class whose normalised name is not present in merged_names (e.g. it was
    dropped via --drop-classes) is simply omitted from the returned map, so
    remap_yolo_labels drops its boxes.
    """
    lower_pos = {n.lower(): i for i, n in enumerate(merged_names)}
    out = {}
    for i, name in enumerate(src_names):
        norm = normalise_class(name, merged_names)
        pos = lower_pos.get(norm.lower())
        if pos is not None:
            out[i] = pos
    return out


def remap_yolo_labels(labels_dir, out_dir, index_map):
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for txt in labels_dir.glob('*.txt'):
        lines_out = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            if cls not in index_map:
                continue
            parts[0] = str(index_map[cls])
            lines_out.append(' '.join(parts))
        target = out_dir / txt.name
        target.write_text('\n'.join(lines_out) + ('\n' if lines_out else ''))
        count += 1
    return count


def read_yaml_names(data_yaml_path):
    data_yaml_path = Path(data_yaml_path)
    text = data_yaml_path.read_text()
    base_dir = data_yaml_path.parent

    names = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('names:'):
            continue
        rest = stripped[len('names:'):].strip()
        if rest.startswith('['):
            # names: ['a', 'b', 'c']
            inner = rest.strip('[]')
            for item in inner.split(','):
                item = item.strip().strip("'\"")
                if item:
                    names.append(item)
        else:
            # block style:
            # names:
            #   0: a
            #   1: b
            block = {}
            for later in lines[idx + 1:]:
                if not later.strip():
                    continue
                if not later.startswith((' ', '\t')):
                    break
                later_stripped = later.strip()
                if ':' not in later_stripped:
                    break
                key, _, value = later_stripped.partition(':')
                key = key.strip()
                value = value.strip().strip("'\"")
                try:
                    block[int(key)] = value
                except ValueError:
                    continue
            for i in sorted(block):
                names.append(block[i])
        break

    def _path_for(key):
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f'{key}:'):
                value = stripped[len(key) + 1:].strip().strip("'\"")
                return (base_dir / value) if value else None
        return None

    return dict(names=names, train=_path_for('train'), val=_path_for('val'))
