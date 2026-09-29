"""Per-camera counting zones: user-drawn polygons that count tracked objects
going in/out of an area and how long they stay, without ever storing an
image -- just numbers, aggregated per day.

Mirrors utils/ignore_areas.py's shape (parse_* validates a JSON-ish config
against the engine's own class vocabulary and raises ValueError on anything
invalid; a per-camera object does the frame-by-frame work) but the geometry
and the state kept per object are different: a zone counts *tracked* objects
(not raw detections) using the FOOTPOINT of their box, and needs a small
state machine per (zone, track) to debounce a track flickering across the
zone edge and to know when to call a track "gone".
"""
import json
import time
from datetime import datetime

MAX_ZONES = 8
MAX_NAME_LEN = 40
MIN_POLYGON_POINTS = 3
MAX_POLYGON_POINTS = 20
METRICS = ('passes', 'dwell')

ENTRY_HYSTERESIS = 0.5   # seconds a footpoint must be continuously inside to confirm entry
EXIT_HYSTERESIS = 0.5    # seconds a footpoint must be continuously outside to confirm exit
PERSON_DWELL_EXIT_HYSTERESIS = 2.0   # people in a "dwell" zone: quicker, they don't park
PERSON_DWELL_LOST_TIMEOUT = 5.0
DWELL_EXIT_HYSTERESIS = 15.0  # same for "dwell" zones: a parked car's footpoint wobbling over the edge isn't leaving
DEFAULT_LOST_TIMEOUT = 3.0    # seconds with no sighting at all -> force-close ("passes" zones)
DWELL_LOST_TIMEOUT = 20.0     # same, but longer for "dwell" zones (a parked car briefly occluded)
STARTUP_GRACE = 15.0          # objects already inside when counting starts are timed, not counted as entries
RESUME_WINDOW = 1800.0        # a "dwell" object lost and re-found at the same spot within this is the same stay
PARKED_CLASSES = frozenset({"car", "truck"})   # only things that park get the lost-and-refound hold
RESUME_DISTANCE = 0.04        # ...when its footpoint is within this (normalised) distance of where it was lost

# A child is reported by the assist detector, not the primary one, but should
# count as a person wherever a zone is watching for people.
CLASS_ALIASES = {'child': 'person'}


def _effective_class(class_name):
    name = (class_name or '').strip().lower()
    return CLASS_ALIASES.get(name, name)


def parse_zones(raw, class_names=None):
    """Validate and normalise a list of counting-zone specs.

    raw: a list of {"id": str, "name": str, "polygon": [[x, y], ...] (3-20
    points, normalised 0-1), "classes": [<name>, ...], "metric": "passes" |
    "dwell"}.
    class_names: the known class-name vocabulary to validate "classes"
    against (e.g. the engine's `class_labels`), or None to skip that check.

    Returns a list of {"id", "name", "polygon": [[x, y], ...], "classes":
    [<lower-case name>, ...], "metric"}. Raises ValueError with a
    human-readable message on any invalid input; raises nothing for `raw`
    being None or [] (both -> []).
    """
    if raw is None or raw == []:
        return []
    if not isinstance(raw, list):
        raise ValueError("count_zones must be a list")
    if len(raw) > MAX_ZONES:
        raise ValueError(f"at most {MAX_ZONES} counting zones are allowed per camera")

    known = {c.lower() for c in class_names} if class_names else None
    zones = []
    seen_ids = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each counting zone must be an object")

        zone_id = entry.get("id")
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError("each counting zone needs a non-empty string id")
        zone_id = zone_id.strip()
        if zone_id in seen_ids:
            raise ValueError(f"duplicate counting zone id: {zone_id}")
        seen_ids.add(zone_id)

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each counting zone needs a non-empty name")
        name = name.strip()
        if len(name) > MAX_NAME_LEN:
            raise ValueError(f"counting zone names must be at most {MAX_NAME_LEN} characters")

        polygon = entry.get("polygon")
        if not isinstance(polygon, list) or not (MIN_POLYGON_POINTS <= len(polygon) <= MAX_POLYGON_POINTS):
            raise ValueError(f"counting zone polygon needs between {MIN_POLYGON_POINTS} and {MAX_POLYGON_POINTS} points")
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each counting zone polygon point must be [x, y]")
            try:
                x, y = (float(v) for v in point)
            except (TypeError, ValueError):
                raise ValueError("counting zone polygon coordinates must be numbers")
            if not (0 <= x <= 1) or not (0 <= y <= 1):
                raise ValueError("counting zone polygon coordinates must be within 0..1")
            points.append([x, y])

        classes = entry.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("each counting zone needs a non-empty list of classes")
        normalised_classes = []
        for c in classes:
            if not isinstance(c, str) or not c.strip():
                raise ValueError("counting zone class names must be non-empty strings")
            lname = c.strip().lower()
            if known is not None and lname not in known:
                raise ValueError(f"unknown class name: {c}")
            if lname not in normalised_classes:
                normalised_classes.append(lname)

        metric = entry.get("metric")
        if metric not in METRICS:
            raise ValueError(f"counting zone metric must be one of {METRICS}")

        zones.append({"id": zone_id, "name": name, "polygon": points,
                       "classes": normalised_classes, "metric": metric})
    return zones


def _point_in_polygon(point, polygon):
    """Standard ray-casting test; polygon is a list of [x, y] (any units, as
    long as they match `point`'s). Boundary points may go either way -- that
    ambiguity doesn't matter here since entry/exit are debounced in time."""
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def footpoint(box):
    """Bottom-centre of a normalised [x1, y1, x2, y2] box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


RIDDEN = frozenset({"bicycle", "motorcycle"})


def _is_riding(person_box, vehicle_box):
    """A person riding shows up as a person box sitting on a bicycle/motorcycle
    box: their footpoint lands inside the vehicle box (padded a little), or
    much of the person box overlaps it."""
    vx1, vy1, vx2, vy2 = vehicle_box
    padx, pady = (vx2 - vx1) * 0.15, (vy2 - vy1) * 0.15
    fx, fy = footpoint(person_box)
    if vx1 - padx <= fx <= vx2 + padx and vy1 - pady <= fy <= vy2 + pady:
        return True
    px1, py1, px2, py2 = person_box
    inter = max(0.0, min(px2, vx2) - max(px1, vx1)) * max(0.0, min(py2, vy2) - max(py1, vy1))
    area = max(1e-9, (px2 - px1) * (py2 - py1))
    return inter / area >= 0.3


def _local_date(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


def _new_class_stats():
    return {"entered": 0, "exited": 0, "dwell_count": 0, "dwell_sum": 0.0, "dwell_max": 0.0}


class ZoneCounter:
    """Per-camera counting-zone state: one instance per camera, holding
    every configured zone's today-so-far stats plus the live (zone, track)
    state machines that produce them.

    `zones` is the list `parse_zones` returns. Stats persist across a
    `reconfigure()` call for any zone id that is unchanged (a config edit for
    zone A must not zero zone B's counts), and across a day boundary they
    reset (see `_maybe_rollover`).
    """

    def __init__(self, zones=None, now=None, startup_grace=STARTUP_GRACE):
        start = now if now is not None else time.time()
        # Tracker ids start fresh whenever the engine starts, so a car parked
        # before a restart would otherwise be counted as a second arrival.
        self._adopt_until = start + startup_grace
        # (zone_id, class) -> [state of a dwell stay whose track was lost,
        # kept so the same object re-found at the same spot resumes it]
        self._limbo = {}
        # track_id -> last time seen riding: a cyclist counts as a bicycle, not
        # as a person walking, even on frames where the bike isn't detected.
        self._riders = {}
        self.zones = {}
        self._stats = {}   # zone_id -> class_name -> _new_class_stats()
        self._state = {}   # (zone_id, track_id) -> dict, see update()
        self._day = _local_date(start)
        self.reconfigure(zones or [])

    def reconfigure(self, zones):
        """Replace the zone config. Stats for a zone id that still exists
        are kept; live (zone, track) state for a zone id that no longer
        exists is dropped (that zone's polygon/classes may have changed
        shape entirely, so there is nothing sound to carry forward)."""
        self.zones = {z["id"]: z for z in zones}
        for key in list(self._state.keys()):
            if key[0] not in self.zones:
                del self._state[key]

    def _maybe_rollover(self, now):
        today = _local_date(now)
        if today != self._day:
            self._day = today
            self._stats = {}
            # Tracks currently inside keep their original entry_time (kept
            # simple, as designed): they just start contributing to the new
            # day's entered/dwell stats only once they exit, same as any
            # other track that happens to still be inside at midnight.

    def _class_stats(self, zone_id, class_name):
        by_class = self._stats.setdefault(zone_id, {})
        return by_class.setdefault(class_name, _new_class_stats())

    def _lost_timeout(self, zone, class_name=None):
        if zone["metric"] != "dwell":
            return DEFAULT_LOST_TIMEOUT
        return DWELL_LOST_TIMEOUT if class_name in PARKED_CLASSES or class_name is None else PERSON_DWELL_LOST_TIMEOUT

    def _exit_hysteresis(self, zone, class_name):
        if zone["metric"] != "dwell":
            return EXIT_HYSTERESIS
        return DWELL_EXIT_HYSTERESIS if class_name in PARKED_CLASSES else PERSON_DWELL_EXIT_HYSTERESIS

    def update(self, now, tracks):
        """tracks: iterable of (track_id, class_name, (x1, y1, x2, y2)),
        box normalised 0-1 in the frame. Call once per processed frame."""
        self._maybe_rollover(now)
        tracks = list(tracks)
        vehicles = [box for _tid, cls, box in tracks if cls in RIDDEN]
        for track_id, cls, box in tracks:
            if _effective_class(cls) == "person" and any(_is_riding(box, v) for v in vehicles):
                self._riders[track_id] = now
        for track_id, seen in list(self._riders.items()):
            if now - seen > 60:
                del self._riders[track_id]
        tracks = [(tid, "rider" if (_effective_class(cls) == "person" and tid in self._riders) else cls, box)
                  for tid, cls, box in tracks]
        seen_keys = set()

        for zone_id, zone in self.zones.items():
            classes = zone["classes"]
            polygon = zone["polygon"]
            for track_id, class_name, box in tracks:
                eff_class = _effective_class(class_name)
                if eff_class not in classes:
                    continue
                key = (zone_id, track_id)
                seen_keys.add(key)
                state = self._state.get(key)
                if state is None:
                    state = {"confirmed": False, "pending_enter_since": None,
                              "pending_exit_since": None, "entry_time": None,
                              "last_inside_seen": None, "class_name": eff_class}
                    self._state[key] = state
                state["last_seen"] = now
                foot = footpoint(box)
                inside = _point_in_polygon(foot, polygon)
                if inside:
                    state["pending_exit_since"] = None
                    state["last_inside_seen"] = now
                    state["last_foot"] = foot
                    if not state["confirmed"]:
                        if state["pending_enter_since"] is None:
                            state["pending_enter_since"] = now
                        elif now - state["pending_enter_since"] >= ENTRY_HYSTERESIS:
                            state["confirmed"] = True
                            state["entry_time"] = state["pending_enter_since"]
                            resumed = self._resume(zone_id, zone, eff_class, foot, now)
                            if resumed is not None:
                                # Same object as a stay whose track was lost: keep its
                                # arrival time and don't count it again.
                                state["entry_time"] = resumed["entry_time"]
                                state["adopted"] = resumed.get("adopted", False)
                            elif state["pending_enter_since"] < self._adopt_until:
                                state["adopted"] = True   # was here before counting started
                            else:
                                self._class_stats(zone_id, eff_class)["entered"] += 1
                else:
                    state["pending_enter_since"] = None
                    if state["confirmed"]:
                        if state["pending_exit_since"] is None:
                            state["pending_exit_since"] = now
                        elif now - state["pending_exit_since"] >= self._exit_hysteresis(zone, eff_class):
                            self._close(zone_id, key)

        # Tracks not seen at all this frame: force-close once lost long enough.
        for key in list(self._state.keys()):
            if key in seen_keys:
                continue
            zone_id, _track_id = key
            zone = self.zones.get(zone_id)
            if zone is None:
                del self._state[key]
                continue
            state = self._state[key]
            if now - state["last_seen"] >= self._lost_timeout(zone, state["class_name"]):
                if (state["confirmed"] and zone["metric"] == "dwell" and state.get("last_foot")
                        and state["class_name"] in PARKED_CLASSES):
                    # Not closed yet: the detector may just have lost sight of it.
                    del self._state[key]
                    self._limbo.setdefault((zone_id, state["class_name"]), []).append(state)
                elif state["confirmed"]:
                    self._close(zone_id, key)
                else:
                    del self._state[key]
        self._expire_limbo(now)

    def _resume(self, zone_id, zone, class_name, foot, now):
        """Pop and return a lost stay of this class whose last footpoint was
        near `foot`, if one is waiting for this zone; else None."""
        if zone["metric"] != "dwell" or class_name not in PARKED_CLASSES:
            return None
        waiting = self._limbo.get((zone_id, class_name)) or []
        best = None
        for state in waiting:
            fx, fy = state["last_foot"]
            dist = ((fx - foot[0]) ** 2 + (fy - foot[1]) ** 2) ** 0.5
            if dist <= RESUME_DISTANCE and (best is None or dist < best[0]):
                best = (dist, state)
        if best is None:
            return None
        waiting.remove(best[1])
        return best[1]

    def _expire_limbo(self, now):
        """A lost stay nobody resumed within RESUME_WINDOW really ended when it
        was last seen inside: record it then."""
        for (zone_id, _cls), waiting in list(self._limbo.items()):
            for state in list(waiting):
                if now - state["last_seen"] >= RESUME_WINDOW or zone_id not in self.zones:
                    waiting.remove(state)
                    if zone_id in self.zones:
                        self._record_exit(zone_id, state)

    def _close(self, zone_id, key):
        self._record_exit(zone_id, self._state.pop(key))

    def _record_exit(self, zone_id, state):
        stats = self._class_stats(zone_id, state["class_name"])
        if state.get("adopted"):
            # Arrived before counting started: its true stay is unknown, so it
            # neither counts as an entry nor skews the average.
            return
        stats["exited"] += 1
        dwell = (state["last_inside_seen"] or state["entry_time"]) - state["entry_time"]
        stats["dwell_count"] += 1
        stats["dwell_sum"] += dwell
        stats["dwell_max"] = max(stats["dwell_max"], dwell)

    def dwell_timers(self, now=None):
        """{track_id: (seconds inside, adopted)} for tracks currently inside a
        "dwell" zone -- for labelling boxes in the live view. Safe to call from
        another thread (reads a snapshot of the state)."""
        now = now if now is not None else time.time()
        timers = {}
        for (zone_id, track_id), state in list(self._state.items()):
            zone = self.zones.get(zone_id)
            if zone is None or zone["metric"] != "dwell" or not state.get("confirmed") or state.get("entry_time") is None:
                continue
            seconds = max(0.0, now - state["entry_time"])
            if track_id not in timers or seconds > timers[track_id][0]:
                timers[track_id] = (seconds, bool(state.get("adopted")))
        return timers

    def snapshot(self, now=None):
        """JSON-ready per-zone stats for the current local day, plus who is
        inside right now. Does not mutate state beyond a day rollover check."""
        now = now if now is not None else time.time()
        self._maybe_rollover(now)
        zones_out = []
        for zone_id, zone in self.zones.items():
            by_class = {}
            for class_name in zone["classes"]:
                stats = self._stats.get(zone_id, {}).get(class_name, _new_class_stats())
                dwell_avg = (stats["dwell_sum"] / stats["dwell_count"]) if stats["dwell_count"] else 0.0
                by_class[class_name] = {
                    "entered": stats["entered"], "exited": stats["exited"],
                    "dwell_count": stats["dwell_count"], "dwell_avg": dwell_avg,
                    "dwell_max": stats["dwell_max"],
                }
            inside_now = [
                {"track_id": track_id, "class": state["class_name"], "since": state["entry_time"],
                 "adopted": bool(state.get("adopted"))}
                for (zid, track_id), state in self._state.items()
                if zid == zone_id and state["confirmed"]
            ]
            # A stay whose track is momentarily lost is still there as far as
            # anyone watching can tell (a parked car the detector blinked on).
            for (zid, cls), waiting in self._limbo.items():
                if zid == zone_id:
                    inside_now += [{"track_id": None, "class": cls, "since": st["entry_time"],
                                    "adopted": bool(st.get("adopted"))} for st in waiting]
            zones_out.append({
                "id": zone_id, "name": zone["name"], "metric": zone["metric"],
                "classes": list(zone["classes"]), "stats": by_class, "inside_now": inside_now,
            })
        return zones_out

    def to_dict(self):
        """Today's aggregates only (no inside_now, no live track state) --
        what gets persisted to disk."""
        return {"date": self._day, "zones": {
            zone_id: {cls: dict(stats) for cls, stats in by_class.items()}
            for zone_id, by_class in self._stats.items()
        }}

    def load_dict(self, data):
        """Merge a previously-saved to_dict() payload in, if it is for
        today; a stale (yesterday-or-older) file is simply ignored, same as
        never having been loaded."""
        if not data or data.get("date") != self._day:
            return
        for zone_id, by_class in (data.get("zones") or {}).items():
            for class_name, stats in by_class.items():
                target = self._class_stats(zone_id, class_name)
                target["entered"] = stats.get("entered", 0)
                target["exited"] = stats.get("exited", 0)
                target["dwell_count"] = stats.get("dwell_count", 0)
                target["dwell_sum"] = stats.get("dwell_sum", 0.0)
                target["dwell_max"] = stats.get("dwell_max", 0.0)

    def save(self, path):
        path = _as_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    def load(self, path):
        path = _as_path(path)
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            return
        self.load_dict(data)


def _as_path(path):
    from pathlib import Path
    return path if hasattr(path, 'parent') else Path(path)
