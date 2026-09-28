import numpy as np
from ocsort_tracker import ocsort


def track(dx, frames=20, w=40, h=55, cls=1):
    t = ocsort.OCSort(max_age=100)
    shown, ids = 0, set()
    for i in range(frames):
        x = 100 + dx * i
        out = t.update(np.array([[x, 300, x + w, 300 + h, 0.62, cls]], dtype=float), 0.5)
        if out:
            shown += 1
            ids.update(int(o.track_id) for o in out)
    return shown, ids


def test_fast_bike_keeps_one_track():
    # ~40 px per detection at 10 detections/s: a bike crossing a 1280 px frame in ~3 s
    for dx in (25, 40, 60):
        shown, ids = track(dx)
        assert shown >= 18 and len(ids) == 1, (dx, shown, ids)


def test_slow_objects_unchanged():
    shown, ids = track(5)
    assert shown == 20 and len(ids) == 1


def test_distance_match_needs_same_class_and_size():
    dets = np.array([[140, 300, 180, 355]], dtype=float)
    trk = np.array([[100, 300, 140, 355]], dtype=float)
    assert ocsort.distance_associate(dets, np.array([1]), trk, np.array([1]), np.array([1])) == [(0, 0)]
    assert ocsort.distance_associate(dets, np.array([1]), trk, np.array([2]), np.array([1])) == []
    big = np.array([[100, 300, 260, 520]], dtype=float)
    assert ocsort.distance_associate(dets, np.array([1]), big, np.array([1]), np.array([1])) == []
    far = np.array([[700, 300, 740, 355]], dtype=float)
    assert ocsort.distance_associate(dets, np.array([1]), far, np.array([1]), np.array([1])) == []
    assert ocsort.distance_associate(dets, np.array([1]), trk, np.array([1]), np.array([9])) == []   # stale track


def test_two_people_passing_keep_their_own_tracks():
    t = ocsort.OCSort(max_age=100)
    ids_a, ids_b = set(), set()
    for i in range(20):
        a, b = 100 + 30 * i, 900 - 30 * i
        out = t.update(np.array([[a, 300, a + 40, 355, 0.7, 0], [b, 420, b + 40, 475, 0.7, 0]], dtype=float), 0.5)
        for o in out:
            (ids_a if o.tlwh[1] < 400 else ids_b).add(int(o.track_id))
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b
