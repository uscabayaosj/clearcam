"""Unit coverage for live-only mode (record_video=False)."""
import ast
import json
import tempfile
import time
import unittest
from pathlib import Path

from utils.live_journal import LiveJournal
from utils.local_descriptions import LocalDescriptions, trigger_crop
from utils import corrections, summaries


class LiveJournalTests(unittest.TestCase):
    def test_add_returns_id_and_list_is_newest_first(self):
        journal = LiveJournal(maxlen=10)
        first = journal.add(dict(id=1, cam_name='front', captured_at=1.0))
        second = journal.add(dict(id=2, cam_name='front', captured_at=2.0))
        self.assertEqual([e['id'] for e in journal.list()], [second, first])

    def test_get_and_update(self):
        journal = LiveJournal(maxlen=10)
        event_id = journal.add(dict(id=99, cam_name='back', captured_at=1.0, description=None))
        self.assertEqual(journal.get(event_id)['cam_name'], 'back')
        updated = journal.update(event_id, description='A person walks by.')
        self.assertEqual(updated['description'], 'A person walks by.')
        self.assertEqual(journal.get(event_id)['description'], 'A person walks by.')
        self.assertIsNone(journal.update('missing', description='x'))

    def test_filters_by_cam_and_alerts_only(self):
        journal = LiveJournal(maxlen=10)
        journal.add(dict(id=1, cam_name='front', captured_at=1.0, is_notif=True))
        journal.add(dict(id=2, cam_name='back', captured_at=2.0, is_notif=False))
        journal.add(dict(id=3, cam_name='front', captured_at=3.0, is_notif=False))
        self.assertEqual([e['id'] for e in journal.list(cam='front')], ['3', '1'])
        self.assertEqual([e['id'] for e in journal.list(alerts_only=True)], ['1'])

    def test_maxlen_evicts_oldest_and_prunes_index(self):
        journal = LiveJournal(maxlen=3)
        ids = [journal.add(dict(id=i, cam_name='front', captured_at=float(i))) for i in range(5)]
        self.assertEqual(len(journal.list(count=100)), 3)
        self.assertIsNone(journal.get(ids[0]))
        self.assertIsNone(journal.get(ids[1]))
        self.assertIsNotNone(journal.get(ids[4]))

    def test_all_since(self):
        journal = LiveJournal(maxlen=10)
        journal.add(dict(id=1, cam_name='front', captured_at=10.0))
        journal.add(dict(id=2, cam_name='front', captured_at=20.0))
        self.assertEqual(len(journal.all_since(15.0)), 1)
        self.assertEqual(len(journal.all_since(0.0)), 2)

    def test_pagination(self):
        journal = LiveJournal(maxlen=10)
        for i in range(5): journal.add(dict(id=i, cam_name='front', captured_at=float(i)))
        page = journal.list(start=1, count=2)
        self.assertEqual(len(page), 2)


class TriggerCropTests(unittest.TestCase):
    def test_returns_ndarray_around_box(self):
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        crop = trigger_crop(frame, (900, 500, 960, 560))
        self.assertIsNotNone(crop)
        self.assertGreaterEqual(min(crop.shape[:2]), 320)

    def test_tiny_box_at_edge_can_return_none_or_small_but_never_raises(self):
        import numpy as np
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        crop = trigger_crop(frame, (0, 0, 2, 2))
        self.assertIsNone(crop)  # smaller than the 64px floor


class SubmitMemoryTests(unittest.TestCase):
    class FakeModel:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): return 'A person stands by the door.'

    class BrokenModel:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): raise RuntimeError('boom')

    def _run(self, model_factory):
        worker = LocalDescriptions(model_factory=model_factory)
        worker.configure(True)
        results = []
        temp_root = Path(tempfile.gettempdir()) / 'clearcam-live'
        before = set(temp_root.glob('*')) if temp_root.is_dir() else set()
        worker.submit_memory(b'\xff\xd8\xff\xdb fake jpeg bytes', 'Front', False, 'Describe.', results.append)
        deadline = time.monotonic() + 5
        while worker.jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.01)
        after = set(temp_root.glob('*')) if temp_root.is_dir() else set()
        return results, after - before

    def test_calls_on_result_and_deletes_temp_file(self):
        results, leftovers = self._run(self.FakeModel)
        self.assertEqual(results, ['A person stands by the door.'])
        self.assertEqual(leftovers, set())

    def test_deletes_temp_file_even_when_model_raises(self):
        with self.assertLogs('utils.local_descriptions', level='ERROR'):
            results, leftovers = self._run(self.BrokenModel)
        self.assertEqual(results, [None])
        self.assertEqual(leftovers, set())

    def test_disabled_calls_on_result_with_none_and_does_not_queue(self):
        worker = LocalDescriptions(model_factory=self.FakeModel)
        results = []
        self.assertFalse(worker.submit_memory(b'bytes', 'Front', False, 'Describe.', results.append))
        self.assertEqual(results, [None])


class CollectEventsTests(unittest.TestCase):
    def test_groups_facts_from_plain_dicts(self):
        events = [
            dict(time=10.0, camera='front', people=['Alice'], description='Alice walks in.'),
            dict(time=20.0, camera='back', people=[], description=None),
            dict(time=99999.0, camera='front', people=[], description=None),  # outside window
        ]
        facts = summaries.collect_events(events, 0.0, 100.0)
        self.assertEqual(len(facts['events']), 2)
        self.assertEqual(facts['cameras'], {'front': 1, 'back': 1})
        self.assertEqual(facts['people'], {'Alice': 1})
        self.assertEqual(facts['unrecognized'], 1)


class RecordCorrectionBytesTests(unittest.TestCase):
    def test_writes_image_and_jsonl_row(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            detections = [dict(box=[1, 2, 3, 4], score=0.9, cls=0, label='person', trigger=True)]
            entry = corrections.record_correction_bytes(
                root, b'jpegbytes', '123_notif.jpg', detections, 1920, 1080,
                'wrong_label', 'bicycle', 'front', crop_bytes=b'cropbytes')
            self.assertEqual(entry['label'], 'bicycle')
            images = list((root / 'corrections' / 'images').iterdir())
            self.assertEqual(len(images), 2)  # image + crop
            rows = (root / 'corrections' / 'corrections.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 1)
            stored = json.loads(rows[0])
            self.assertEqual(stored['verdict'], 'wrong_label')
            self.assertEqual(stored['camera'], 'front')
            self.assertEqual(len(corrections.load_corrections(root)), 1)

    def test_rejects_bad_verdicts(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                corrections.record_correction_bytes(folder, b'x', 'a.jpg', [], 10, 10, 'nonsense', None, 'front')
            with self.assertRaises(ValueError):
                corrections.record_correction_bytes(folder, b'x', 'a.jpg', [], 10, 10, 'wrong_label', None, 'front')


def _load_counts_helpers():
    source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
    wanted = {'counts_phrase', 'caption_due'}
    funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<counts_helpers>', 'exec'), namespace)
    return namespace['counts_phrase'], namespace['caption_due']


class CountsPhraseAndCaptionDueTests(unittest.TestCase):
    def setUp(self):
        self.counts_phrase, self.caption_due = _load_counts_helpers()

    def test_counts_phrase_examples(self):
        self.assertEqual(self.counts_phrase({}), 'no detected objects')
        self.assertEqual(self.counts_phrase({'person': 1}), '1 person')
        self.assertEqual(self.counts_phrase({'car': 2, 'person': 1}), '2 cars and 1 person')
        self.assertEqual(self.counts_phrase({'box': 2}), '2 boxes')

    def test_caption_due_no_prior_caption(self):
        self.assertTrue(self.caption_due({}, {'person': 1}, None, 100.0))

    def test_caption_due_change_needs_30s(self):
        self.assertFalse(self.caption_due({}, {'person': 1}, 90.0, 100.0))   # only 10s old
        self.assertTrue(self.caption_due({}, {'person': 1}, 60.0, 100.0))    # 40s old, changed

    def test_caption_due_unchanged_stays_false_until_120s(self):
        self.assertFalse(self.caption_due({'person': 1}, {'person': 1}, 50.0, 100.0))
        self.assertTrue(self.caption_due({'person': 1}, {'person': 1}, -30.0, 100.0))  # 130s old


if __name__ == '__main__':
    unittest.main()
