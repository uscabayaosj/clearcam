import io
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import collect_frames as cf


# --------------------------------------------------------------- first_jpeg

def test_first_jpeg_valid_frame():
    header = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n'
    data = b'\xff\xd8\x00\x00'
    assert cf.first_jpeg(io.BytesIO(header + data)) == data


def test_first_jpeg_garbage_stream_no_header_terminator():
    # over 4096 bytes with no \r\n\r\n anywhere -> header-parsing bails out
    assert cf.first_jpeg(io.BytesIO(b'x' * 5000)) is None


def test_first_jpeg_oversized_content_length_rejected():
    header = b'Content-Length: 999999999\r\n\r\n'
    assert cf.first_jpeg(io.BytesIO(header)) is None


def test_first_jpeg_zero_or_negative_length_rejected():
    header = b'Content-Length: 0\r\n\r\n'
    assert cf.first_jpeg(io.BytesIO(header)) is None


def test_first_jpeg_missing_content_length_header():
    assert cf.first_jpeg(io.BytesIO(b'--frame\r\n\r\n')) is None


def test_first_jpeg_bad_magic_bytes_rejected():
    header = b'Content-Length: 4\r\n\r\n'
    assert cf.first_jpeg(io.BytesIO(header + b'abcd')) is None


def test_first_jpeg_truncated_stream_returns_none():
    header = b'Content-Length: 10\r\n\r\n'
    assert cf.first_jpeg(io.BytesIO(header + b'\xff\xd8\x00\x00')) is None


def test_first_jpeg_empty_stream_returns_none():
    assert cf.first_jpeg(io.BytesIO(b'')) is None


# -------------------------------------------------------------- pick_varied

def _hourly_frames():
    """9 frames across 3 hours; within an hour, values are close together
    (< min_diff apart); across hours they're far apart."""
    values = {}
    frames = []
    for hour, base in [('00', 0), ('01', 50), ('02', 100)]:
        for i in range(3):
            name = f'20250101-{hour}{i:02d}00.jpg'
            values[name] = base + i
            frames.append(Path(name))
    return frames, values


def test_pick_varied_spreads_across_hours():
    frames, values = _hourly_frames()
    thumb = lambda p: [values[p.name]]
    picked = cf.pick_varied(frames, limit=10, min_diff=6.0, thumb=thumb)
    hours = [p.name[:11] for p in picked]
    assert sorted(set(hours)) == ['20250101-00', '20250101-01', '20250101-02']


def test_pick_varied_enforces_min_diff_within_an_hour():
    frames, values = _hourly_frames()
    thumb = lambda p: [values[p.name]]
    picked = cf.pick_varied(frames, limit=10, min_diff=6.0, thumb=thumb)
    # within each hour the 3 candidate values are only 1 apart (< min_diff),
    # so only the first-visited frame from each hour should survive
    counts = Counter(p.name[:11] for p in picked)
    assert set(counts.values()) == {1}
    assert len(picked) == 3


def test_pick_varied_respects_limit():
    frames, values = _hourly_frames()
    thumb = lambda p: [values[p.name]]
    picked = cf.pick_varied(frames, limit=2, min_diff=6.0, thumb=thumb)
    assert len(picked) == 2


def test_pick_varied_returns_sorted_output():
    frames, values = _hourly_frames()
    thumb = lambda p: [values[p.name]]
    picked = cf.pick_varied(list(reversed(frames)), limit=10, min_diff=6.0, thumb=thumb)
    assert picked == sorted(picked)


# --------------------------------------------------------------- split_for

def test_split_for_deterministic():
    assert cf.split_for('20250101-000000.jpg', 'front-door') == cf.split_for('20250101-000000.jpg', 'front-door')


def test_split_for_only_three_outcomes():
    outcomes = {cf.split_for(f'{i}.jpg', 'cam') for i in range(500)}
    assert outcomes <= {'train', 'valid', 'holdout'}


def test_split_for_roughly_70_15_15_over_many_names():
    counts = Counter(cf.split_for(f'{i}.jpg', 'front-door') for i in range(4000))
    total = sum(counts.values())
    assert abs(counts['train'] / total - 0.70) < 0.03
    assert abs(counts['valid'] / total - 0.15) < 0.03
    assert abs(counts['holdout'] / total - 0.15) < 0.03


def test_split_for_varies_with_camera():
    # same filename, different cameras -> not forced into lockstep splits
    per_camera = {cf.split_for('same-name.jpg', f'cam{i}') for i in range(20)}
    assert len(per_camera) > 1


def test_split_by_hour_holds_back_whole_hours():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / 'script'))
    import collect_frames as cf
    assert cf.split_by_hour('20260928-010203.jpg') == 'holdout'
    assert cf.split_by_hour('20260928-030000.jpg') == 'valid'
    assert cf.split_by_hour('20260928-220000.jpg') == 'train'
    assert {cf.split_by_hour(f'20260928-01{m:02d}00.jpg') for m in range(60)} == {'holdout'}


def test_moving_now_and_window():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / 'script'))
    import collect_frames as cf
    scene = {'cameras': {'a': {'counts': {'car': 3}}, 'b': {'counts': {'person': 1, 'car': 1}},
                         'c': {'counts': {'bicycle': 0}}, 'd': {'counts': {}}}}
    assert cf.moving_now(scene) == ['b']
    assert cf.in_window(7, (7, 20)) and not cf.in_window(20, (7, 20)) and cf.in_window(3, None)
