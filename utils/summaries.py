"""Recurring local AI summaries of a monitoring period.

Aggregates the event sidecars already on disk (times, cameras, household
names, per-event descriptions) into facts; a deterministic template turns the
facts into an honest summary, and the local model may rewrite that same
structure more fluently. Nothing here invents events: the model only ever
sees, and may only restate, what the recorded facts contain.
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import time

from utils.household import read_people
from utils.local_descriptions import read_description
from utils.recording_timeline import event_timing

DEFAULT_TIME = '21:00'
MAX_DESCRIPTIONS_IN_PROMPT = 20


def collect_window(cameras_root, start_ts, end_ts):
    """Facts about every event between start_ts and end_ts (wall clock)."""
    events = []
    for image in Path(cameras_root).glob('*/event_images/*/*.jpg'):
        try:
            captured = event_timing(image).get('captured_at')
        except Exception:
            captured = None
        if captured is None: captured = image.stat().st_mtime
        if not (start_ts <= captured < end_ts): continue
        events.append(dict(
            time=captured,
            camera=image.parts[-4],
            people=read_people(image) or [],
            description=read_description(image),
        ))
    events.sort(key=lambda e: e['time'])
    cameras = Counter(e['camera'] for e in events)
    people = Counter(name for e in events for name in e['people'])
    unrecognized = sum(1 for e in events if e['people'] == [])
    hours = sorted({datetime.fromtimestamp(e['time']).hour for e in events})
    return dict(start=start_ts, end=end_ts, events=events, cameras=dict(cameras),
                people=dict(people), unrecognized=unrecognized, active_hours=hours)


def _plural(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _span_label(start_ts, end_ts):
    start, end = datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts)
    if start.date() == end.date():
        return f"{start.strftime('%a %d %b, %H:%M')}–{end.strftime('%H:%M')}"
    return f"{start.strftime('%a %H:%M')} – {end.strftime('%a %H:%M')}"


def quiet_ranges(active_hours, start_ts, end_ts):
    """Contiguous inactive hour ranges inside the window, largest first."""
    window_hours = set()
    cursor = start_ts
    while cursor < end_ts:
        window_hours.add(datetime.fromtimestamp(cursor).hour)
        cursor += 3600
    inactive = sorted(window_hours - set(active_hours))
    if not inactive: return []
    ranges, run = [], [inactive[0]]
    for hour in inactive[1:]:
        if hour == run[-1] + 1: run.append(hour)
        else: ranges.append(run); run = [hour]
    ranges.append(run)
    return sorted((f"{r[0]:02d}:00–{(r[-1] + 1) % 24:02d}:00" for r in ranges if len(r) >= 2),
                  key=len, reverse=True)


def deterministic_summary(facts):
    """The honest template. Also the shape the model is asked to preserve."""
    events = facts['events']
    if not events:
        return f"No activity was detected between {_span_label(facts['start'], facts['end'])}. All cameras stayed quiet."
    lines = [f"{_plural(len(events), 'event')} across {_plural(len(facts['cameras']), 'camera')} "
             f"({_span_label(facts['start'], facts['end'])})."]
    for name, count in sorted(facts['people'].items(), key=lambda kv: -kv[1]):
        seen_on = sorted({e['camera'] for e in events if name in e['people']})
        lines.append(f"{name} was seen {_plural(count, 'time')} ({', '.join(seen_on)}).")
    if facts['unrecognized']:
        # "Not recognized" is what the data supports; a face may simply not be visible.
        lines.append(f"A person was seen {_plural(facts['unrecognized'], 'time')} without being recognized.")
    busiest = max(facts['cameras'].items(), key=lambda kv: kv[1])
    if len(facts['cameras']) > 1:
        lines.append(f"Busiest camera: {busiest[0]} ({_plural(busiest[1], 'event')}).")
    quiet = quiet_ranges(facts['active_hours'], facts['start'], facts['end'])
    if quiet:
        lines.append(f"Quiet hours: {', '.join(quiet[:2])}.")
    return ' '.join(lines)


def build_prompt(facts):
    described = [e for e in facts['events'] if e['description']][:MAX_DESCRIPTIONS_IN_PROMPT]
    observations = '\n'.join(
        f"- {datetime.fromtimestamp(e['time']).strftime('%H:%M')} {e['camera']}"
        f"{' (' + ', '.join(e['people']) + ')' if e['people'] else ''}: {e['description']}"
        for e in described)
    return (
        "You summarize a home camera's monitoring period for its owner. Use ONLY the facts below; "
        "never invent events, people, or details. Group repeated sightings of the same person or "
        "object and state how many times each was seen, e.g. \"A man in a black shirt was seen 4 "
        "times\". Mention recognized household names first with their counts, then unrecognized "
        "people, then notable objects or actions, then quiet periods. At most 5 plain sentences, "
        "under 80 words, no headings, no advice, and never repeat a sentence.\n\n"
        f"Baseline summary (keep its numbers exactly): {deterministic_summary(facts)}\n\n"
        f"Individual observations:\n{observations if observations else '- none recorded'}\n\nSummary:"
    )


def acceptable_summary(text):
    """Reject degenerate model output (repetition loops, fragments).

    Small models can loop under greedy decoding; a looping summary is worse
    than the deterministic template, so the gate is strict.
    """
    if not text or len(text.strip()) < 20: return False
    sentences = [s.strip().lower() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    if not sentences: return False
    counts = Counter(sentences)
    if counts.most_common(1)[0][1] >= 3: return False
    if len(counts) / len(sentences) < 0.6: return False
    return True


def parse_daily_time(value):
    match = re.fullmatch(r'([01]?\d|2[0-3]):([0-5]\d)', str(value or '').strip())
    if not match: raise ValueError('Time must be HH:MM')
    return int(match.group(1)), int(match.group(2))


def is_due(config, now=None):
    """A daily summary is due once per day at the configured time."""
    if not config or not config.get('enabled'): return False
    now = time.time() if now is None else now
    hour, minute = parse_daily_time(config.get('time', DEFAULT_TIME))
    today_run = datetime.fromtimestamp(now).replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()
    return now >= today_run and config.get('last_run', 0) < today_run


def summary_dir(data_root):
    return Path(data_root) / 'summaries'


def write_summary(data_root, payload):
    target = summary_dir(data_root)
    target.mkdir(parents=True, exist_ok=True)
    path = target / (datetime.fromtimestamp(payload['end']).strftime('%Y-%m-%d-%H%M') + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)
    return path


def recent_summaries(data_root, count=7):
    target = summary_dir(data_root)
    if not target.is_dir(): return []
    entries = []
    for path in sorted(target.glob('*.json'), reverse=True)[:count]:
        try:
            entries.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return entries
