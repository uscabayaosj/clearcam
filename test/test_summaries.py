import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from utils import summaries


def make_event(root, camera, when, people=None, description=None):
    folder = root / camera / 'event_images' / when.strftime('%Y-%m-%d')
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / f'{int(when.timestamp() * 1e9)}_notif.jpg'
    image.write_bytes(b'jpg')
    image.with_suffix('.event.json').write_text(json.dumps(dict(captured_at=when.timestamp())))
    if people is not None:
        image.with_suffix('.people.json').write_text(json.dumps(dict(people=people)))
    if description:
        image.with_suffix('.description.json').write_text(json.dumps(dict(description=description, generated=True)))
    return image


class SummariesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cameras = self.root / 'cameras'
        self.noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    def window(self):
        return (self.noon - timedelta(hours=12)).timestamp(), (self.noon + timedelta(hours=12)).timestamp()

    def test_collect_aggregates_people_cameras_and_descriptions(self):
        make_event(self.cameras, 'front', self.noon, people=['Ulysses'], description='A man walks by.')
        make_event(self.cameras, 'front', self.noon + timedelta(hours=1), people=['Ulysses'])
        make_event(self.cameras, 'back', self.noon + timedelta(hours=2), people=[], description='A cat sits.')
        make_event(self.cameras, 'back', self.noon - timedelta(days=2))  # outside window
        start, end = self.window()
        facts = summaries.collect_window(self.cameras, start, end)
        self.assertEqual(len(facts['events']), 3)
        self.assertEqual(facts['people'], {'Ulysses': 2})
        self.assertEqual(facts['unrecognized'], 1)
        self.assertEqual(facts['cameras'], {'front': 2, 'back': 1})

    def test_deterministic_summary_counts_and_quiet(self):
        make_event(self.cameras, 'front', self.noon, people=['Ulysses'])
        start, end = self.window()
        facts = summaries.collect_window(self.cameras, start, end)
        text = summaries.deterministic_summary(facts)
        self.assertIn('Ulysses was seen 1 time', text)
        self.assertIn('1 event across 1 camera', text)
        self.assertIn('Quiet hours', text)

    def test_empty_window_is_honest(self):
        start, end = self.window()
        facts = summaries.collect_window(self.cameras, start, end)
        self.assertIn('No activity was detected', summaries.deterministic_summary(facts))

    def test_prompt_contains_baseline_and_observations_only(self):
        make_event(self.cameras, 'front', self.noon, people=['Ana'], description='A red van stops.')
        start, end = self.window()
        prompt = summaries.build_prompt(summaries.collect_window(self.cameras, start, end))
        self.assertIn('A red van stops.', prompt)
        self.assertIn('never invent', prompt)
        self.assertIn('Ana was seen 1 time', prompt)

    def test_is_due_once_per_day(self):
        now = datetime.now().replace(hour=21, minute=30).timestamp()
        config = dict(enabled=True, time='21:00', last_run=0)
        self.assertTrue(summaries.is_due(config, now))
        config['last_run'] = now
        self.assertFalse(summaries.is_due(config, now + 60))
        self.assertFalse(summaries.is_due(dict(enabled=False, time='21:00', last_run=0), now))
        before = datetime.now().replace(hour=20, minute=0).timestamp()
        self.assertFalse(summaries.is_due(dict(enabled=True, time='21:00', last_run=0), before))
        with self.assertRaises(ValueError): summaries.parse_daily_time('25:99')

    def test_quality_gate_rejects_degenerate_output(self):
        self.assertFalse(summaries.acceptable_summary('A man was seen 2 times. ' * 10))
        self.assertFalse(summaries.acceptable_summary(''))
        self.assertFalse(summaries.acceptable_summary('Short.'))
        self.assertTrue(summaries.acceptable_summary(
            '64 events across 2 cameras. Ulysses was seen 3 times. A cat was seen once. Quiet overnight.'))

    def test_write_and_read_summaries(self):
        payload = dict(start=time.time() - 3600, end=time.time(), summary='All quiet.', generated=False, events=0, model='template')
        summaries.write_summary(self.root, payload)
        recent = summaries.recent_summaries(self.root)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]['summary'], 'All quiet.')


if __name__ == '__main__':
    unittest.main()
