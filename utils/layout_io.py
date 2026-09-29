"""Copy per-camera layout (counting zones, ignore areas, alert zone) between Macs.

The exported file holds only geometry that is meaningful on any Mac, keyed by
camera name. It never carries camera URLs, credentials, thresholds or any other
setting. Importing replaces exactly those three settings for cameras that exist
on this Mac and leaves every other settings key alone.
"""
import json
from datetime import datetime
from pathlib import Path

from utils import count_zones as count_zones_mod
from utils import ignore_areas as ignore_areas_mod

FORMAT_VERSION = 1
MIN_COORDS = 3
MAX_COORDS = 50
MAX_CAMERAS = 100


def _serialise_areas(parsed):
    return [({"box": a["box"], "classes": sorted(a["classes"])} if a["classes"] else {"box": a["box"]})
            for a in parsed]


def parse_alert_zone(raw):
    """Validate an alert zone: None or {"coords": [[x, y], ...], "outside": bool}.
    Returns the normalised dict, or None. Raises ValueError."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("alert_zone must be an object or null")
    coords = raw.get("coords")
    if not isinstance(coords, list) or not (MIN_COORDS <= len(coords) <= MAX_COORDS):
        raise ValueError(f"alert zone needs between {MIN_COORDS} and {MAX_COORDS} points")
    points = []
    for point in coords:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each alert zone point must be [x, y]")
        if any(isinstance(v, bool) for v in point):
            raise ValueError("alert zone coordinates must be numbers")
        try:
            x, y = (float(v) for v in point)
        except (TypeError, ValueError):
            raise ValueError("alert zone coordinates must be numbers")
        if not (0 <= x <= 1) or not (0 <= y <= 1):
            raise ValueError("alert zone coordinates must be within 0..1")
        points.append([x, y])
    outside = raw.get("outside", False)
    if outside is None:
        outside = False
    if not isinstance(outside, bool):
        raise ValueError("alert zone outside must be true or false")
    return {"coords": points, "outside": outside}


def _entry_from_settings(settings, class_names=None):
    settings = settings or {}
    zones = count_zones_mod.parse_zones(settings.get("count_zones"), class_names)
    areas = _serialise_areas(ignore_areas_mod.parse_areas(settings.get("ignore_areas"), class_names))
    alert = None
    if settings.get("coords"):
        alert = parse_alert_zone({"coords": settings["coords"], "outside": bool(settings.get("outside"))})
    return {"count_zones": zones, "ignore_areas": areas, "alert_zone": alert}


def build_layout(settings_by_camera, now=None, class_names=None):
    """settings_by_camera: {camera name: stored settings dict}. Returns the
    export document. A camera whose stored data no longer validates is
    exported with what does (its other parts are never dropped silently for
    another camera's sake)."""
    now = now or datetime.now().astimezone()
    cameras = {}
    for name in sorted(settings_by_camera):
        settings = settings_by_camera[name] or {}
        try:
            cameras[name] = _entry_from_settings(settings, class_names)
        except ValueError:
            entry = {"count_zones": [], "ignore_areas": [], "alert_zone": None}
            for key, fn in (("count_zones", lambda: count_zones_mod.parse_zones(settings.get("count_zones"), class_names)),
                            ("ignore_areas", lambda: _serialise_areas(ignore_areas_mod.parse_areas(settings.get("ignore_areas"), class_names))),
                            ("alert_zone", lambda: _entry_from_settings({"coords": settings.get("coords"), "outside": settings.get("outside")})["alert_zone"])):
                try:
                    entry[key] = fn()
                except ValueError:
                    pass
            cameras[name] = entry
    return {"clearcam_layout": FORMAT_VERSION, "exported_at": now.isoformat(timespec="seconds"), "cameras": cameras}


def parse_layout(data, class_names=None):
    """Validate a whole import document. Returns {name: {"count_zones",
    "ignore_areas", "alert_zone"}} (normalised). Raises ValueError naming the
    camera and the problem; nothing partial is ever returned."""
    if not isinstance(data, dict) or data.get("clearcam_layout") != FORMAT_VERSION:
        raise ValueError("This is not a ClearCam zones file.")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("The zones file has no cameras.")
    if len(cameras) > MAX_CAMERAS:
        raise ValueError(f"The zones file lists more than {MAX_CAMERAS} cameras.")
    parsed = {}
    for name, entry in cameras.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("The zones file has a camera without a name.")
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: expected an object.")
        try:
            zones = count_zones_mod.parse_zones(entry.get("count_zones"), class_names)
            areas = _serialise_areas(ignore_areas_mod.parse_areas(entry.get("ignore_areas"), class_names))
            alert = parse_alert_zone(entry.get("alert_zone"))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name}: {e}")
        parsed[name] = {"count_zones": zones, "ignore_areas": areas, "alert_zone": alert}
    return parsed


def plan_import(parsed, local_cameras):
    """Split a parsed layout by whether each camera exists on this Mac."""
    local = set(local_cameras)
    apply = [n for n in parsed if n in local]
    skip = [n for n in parsed if n not in local]
    counts = {n: {"count_zones": len(parsed[n]["count_zones"]),
                  "ignore_areas": len(parsed[n]["ignore_areas"]),
                  "alert_zone": parsed[n]["alert_zone"] is not None} for n in apply}
    return {"apply": apply, "skip": skip, "counts": counts}


def apply_entry(settings, entry):
    """Return a copy of settings with the three layout settings replaced by
    entry's; every other key is untouched. Empty parts are removed the way
    /edit_settings removes them."""
    out = dict(settings or {})
    for key in ("count_zones", "ignore_areas"):
        if entry[key]:
            out[key] = entry[key]
        else:
            out.pop(key, None)
    alert = entry["alert_zone"]
    if alert:
        out["coords"] = alert["coords"]
        out["outside"] = alert["outside"]
    else:
        out.pop("coords", None)
        out.pop("outside", None)
    return out


def unique_export_path(directory, now=None):
    """~/Downloads/ClearCam zones YYYY-MM-DD.json, then ' 2', ' 3'... -- never overwrites."""
    now = now or datetime.now()
    directory = Path(directory)
    base = f"ClearCam zones {now.strftime('%Y-%m-%d')}"
    path = directory / f"{base}.json"
    n = 2
    while path.exists():
        path = directory / f"{base} {n}.json"
        n += 1
    return path


def write_export(directory, layout, now=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = unique_export_path(directory, now)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
        f.write("\n")
    return path
