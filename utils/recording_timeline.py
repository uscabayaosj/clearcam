"""Separate wall-clock event time from positions in an HLS recording."""
from datetime import datetime
import json
import math
from pathlib import Path


def contained_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError('Path is outside camera storage')
    return target


def read_timeline(playlist):
    playlist = Path(playlist)
    try:
        lines = playlist.read_text().splitlines()
    except OSError:
        return []
    result, offset, duration, wall = [], 0.0, None, None
    for line in lines:
        if line.startswith('#EXT-X-DISCONTINUITY'):
            wall = None
        elif line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
            try:
                wall = datetime.fromisoformat(line.split(':', 1)[1].replace('Z', '+00:00')).timestamp()
            except ValueError:
                wall = None
        elif line.startswith('#EXTINF:'):
            try:
                duration = float(line.split(':', 1)[1].split(',')[0])
            except ValueError:
                duration = None
        elif line and not line.startswith('#') and duration is not None:
            if not math.isfinite(duration) or duration <= 0:
                duration = None
                continue
            try:
                segment = contained_path(playlist.parent, line)
                exists = segment.is_file()
                file_start = segment.stat().st_mtime - duration
                # FFmpeg append_list can synthesize PDT for legacy entries using
                # the new session's clock. Reject those shifted timestamps.
                estimated = wall is None or abs(wall - file_start) > 10
                start = file_start if estimated else wall
            except (ValueError, OSError):
                exists, start, estimated = False, None, True
            result.append(dict(offset=offset, duration=duration, wall=start, exists=exists,
                               estimated=estimated))
            offset += duration
            if wall is not None:
                wall += duration
            duration = None
    return result


def position_at(timeline, wall_time):
    if not isinstance(wall_time, (int, float)) or not math.isfinite(wall_time):
        return None
    for segment in timeline:
        start = segment['wall']
        if segment['exists'] and start is not None and start <= wall_time < start + segment['duration']:
            return segment['offset'] + wall_time - start
    return None


def event_timing(image, timeline=None):
    image = Path(image)
    if image.parent.name == 'video':
        try:
            offset = max(0, float(image.stem.split('_')[0]))
        except ValueError:
            offset = None
        return dict(captured_at=None, playback_offset=offset, time_source='video_offset')
    try:
        data = json.loads(image.with_suffix('.event.json').read_text())
        captured_at = float(data['captured_at'])
        if not math.isfinite(captured_at): raise ValueError('Invalid event time')
        source = 'event'
    except (OSError, ValueError, KeyError, TypeError):
        captured_at, source = image.stat().st_mtime, 'file_time'
    if timeline is None:
        camera = image.parents[2]
        timeline = read_timeline(camera / 'streams' / image.parent.name / 'stream.m3u8')
    # Do not silently seek to unrelated footage when a segment has been removed.
    position = position_at(timeline, captured_at - 5)
    return dict(captured_at=captured_at, playback_offset=position, time_source=source)


def write_event_time(image, captured_at):
    path = Path(image).with_suffix('.event.json')
    temporary = path.with_suffix('.event.tmp')
    temporary.write_text(json.dumps(dict(captured_at=captured_at)))
    temporary.replace(path)


def expired_recording_dirs(camera_root, today):
    """Retention may remove completed days, never the current recording or camera."""
    root = Path(camera_root).resolve()
    candidates = []
    for day in root.glob('*/streams/*'):
        try:
            datetime.strptime(day.name, '%Y-%m-%d')
            if day.name >= today or day.is_symlink() or not day.is_dir(): continue
            if not day.resolve().is_relative_to(root): continue
            candidates.append(day)
        except ValueError:
            continue
    return sorted(candidates, key=lambda day: day.name)


def live_playlist(playlist, window=3):
    """Rolling live-edge view of an append-only event recording.

    Players treat the full day's EVENT playlist as starting from the beginning,
    so the live view needs a short sliding playlist that always points at the
    newest segments while the same segments stay on disk for replay.
    """
    playlist = Path(playlist)
    try:
        lines = playlist.read_text().splitlines()
    except OSError:
        return None
    header, entries, pending, discontinuities = [], [], [], 0
    for line in lines:
        if line.startswith('#EXT-X-ENDLIST'):
            continue
        if line.startswith(('#EXTINF:', '#EXT-X-PROGRAM-DATE-TIME:', '#EXT-X-DISCONTINUITY')):
            pending.append(line)
        elif line and not line.startswith('#'):
            entries.append(pending + [line])
            pending = []
        elif line.startswith(('#EXTM3U', '#EXT-X-VERSION', '#EXT-X-TARGETDURATION')):
            header.append(line)
    if not entries:
        return None

    def duration(entry):
        for tag in entry:
            if tag.startswith('#EXTINF:'):
                try: return float(tag[8:].split(',')[0])
                except ValueError: return 0.0
        return 0.0

    # Apple's HLS engine will not start a live playlist holding less than three
    # target durations of media, and the day's TARGETDURATION is inflated by any
    # single long segment a recorder restart ever produced. So derive the
    # target from the window's own segments and grow the window until it
    # covers three of them.
    kept = entries[-window:]
    while len(kept) < len(entries):
        target = max(1, math.ceil(max(duration(e) for e in kept)))
        if sum(duration(e) for e in kept) >= 3 * target: break
        kept = entries[-(len(kept) + 1):]
    target = max(1, math.ceil(max(duration(e) for e in kept)))
    header = [h for h in header if not h.startswith('#EXT-X-TARGETDURATION')] + [f'#EXT-X-TARGETDURATION:{target}']
    for entry in entries[:-len(kept)]:
        discontinuities += sum(1 for tag in entry if tag.startswith('#EXT-X-DISCONTINUITY'))
    body = [tag for entry in kept for tag in entry]
    # A window that opens mid-session must not lead with a discontinuity marker.
    while body and body[0].startswith('#EXT-X-DISCONTINUITY'):
        discontinuities += 1
        body = body[1:]
    return '\n'.join(header + [
        f'#EXT-X-MEDIA-SEQUENCE:{len(entries) - len(kept)}',
        f'#EXT-X-DISCONTINUITY-SEQUENCE:{discontinuities}',
    ] + body) + '\n'
