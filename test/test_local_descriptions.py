import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from utils.local_descriptions import LocalDescriptions, read_description


class FakeModel:
    def __init__(self, **kwargs):
        pass

    def generate(self, **kwargs):
        assert kwargs['reset'] is True
        assert kwargs['max_tokens'] == 96
        return 'A person stands beside a door.'


class LocalDescriptionTests(unittest.TestCase):
    def test_disabled_does_not_queue(self):
        worker = LocalDescriptions()
        self.assertFalse(worker.submit('/unused.jpg', 'Camera'))
        self.assertEqual(worker.status()['queued'], 0)

    def run_job(self, model):
        worker = LocalDescriptions(model_factory=model)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'event.jpg'
            cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
            worker.configure(True)
            self.assertTrue(worker.submit(image, 'Camera', notify=False))
            deadline = time.monotonic() + 5
            while worker.jobs.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(worker.jobs.unfinished_tasks, 0)
            return worker.status(), read_description(image)

    def test_description_is_persisted(self):
        status, text = self.run_job(FakeModel)
        self.assertEqual(status['state'], 'ready')
        self.assertEqual(text, 'A person stands beside a door.')

    def test_failed_model_does_not_create_description(self):
        class BrokenModel(FakeModel):
            def generate(self, **kwargs):
                raise RuntimeError('Inference failed')
        with self.assertLogs('utils.local_descriptions', level='ERROR'):
            status, text = self.run_job(BrokenModel)
        self.assertEqual(status['state'], 'error')
        self.assertIsNone(text)

    def test_retry_saved_is_bounded_and_does_not_resend_notifications(self):
        worker = LocalDescriptions(model_factory=FakeModel)
        worker.configure(True)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "Owner's camera" / 'event_images' / '2026-08-30'
            folder.mkdir(parents=True)
            for index in range(12):
                (folder / f'{index}.jpg').touch()
            (folder / '11.description.json').write_text('{"description":"Already described"}')
            with patch.object(worker, 'submit', return_value=True) as submit:
                self.assertEqual(worker.retry_saved(directory), 8)
                self.assertEqual(submit.call_count, 8)
                for call in submit.call_args_list:
                    self.assertEqual(call.args[1], "Owner's camera")
                    self.assertFalse(call.kwargs['notify'])
                    self.assertNotEqual(call.args[0].name, '11.jpg')


if __name__ == '__main__':
    unittest.main()


class TriggerCropAndBackfillTests(unittest.TestCase):
    def test_trigger_crop_is_written_around_the_box_and_never_tiny(self):
        import numpy as np, tempfile, cv2
        from pathlib import Path
        from utils.local_descriptions import write_trigger_crop
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            event = Path(folder) / '1_notif.jpg'
            crop = write_trigger_crop(frame, (900, 500, 960, 560), event)   # a 60px box
            self.assertIsNotNone(crop)
            self.assertTrue(str(crop).endswith('.trigger.jpg'))
            h, w = cv2.imread(str(crop)).shape[:2]
            self.assertGreaterEqual(min(h, w), 320)          # enlarged to a useful size
            edge = write_trigger_crop(frame, (1890, 1060, 1919, 1079), event)  # clipped at the corner
            self.assertIsNotNone(edge)

    def test_backfill_only_runs_when_idle_and_rate_limited(self):
        from utils.local_descriptions import LocalDescriptions
        import tempfile
        calls = []
        local = LocalDescriptions(model_factory=lambda **kw: None)
        local.enabled = True
        local.retry_saved = lambda root: calls.append(root) or 1
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(local.backfill_if_idle(root, every=300), 1)
            self.assertEqual(local.backfill_if_idle(root, every=300), 0)   # too soon
            local.last_backfill = 0
            local.jobs.put(('summary', 'p', lambda _: None))                # busy queue
            self.assertEqual(local.backfill_if_idle(root, every=300), 0)
        self.assertEqual(len(calls), 1)

    def test_available_sizes_reflects_packages_on_disk(self):
        import tempfile
        from pathlib import Path
        from detection.coreml_yolo import available_sizes
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'yolo11n.mlpackage').mkdir()
            (Path(folder) / 'yolo11s.mlpackage').mkdir()
            self.assertEqual(available_sizes([None, folder]), ['t', 's'])
            self.assertEqual(available_sizes([None]), [])


class CorrectionsTests(unittest.TestCase):
    def test_detections_then_correction_round_trip(self):
        import json, tempfile
        from pathlib import Path
        from utils import corrections
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            event = root / 'cameras' / 'front' / 'event_images' / '2026-09-01' / '1_notif.jpg'
            event.parent.mkdir(parents=True); event.write_bytes(b'jpg')
            event.with_suffix('.trigger.jpg').write_bytes(b'crop')
            corrections.write_detections(event, [(10, 20, 110, 220, 0.91, 0)], ['person', 'bicycle'], 1920, 1080, trigger_index=0)
            stored = corrections.read_detections(event)
            self.assertEqual(stored['detections'][0]['label'], 'person')
            self.assertTrue(stored['detections'][0]['trigger'])
            entry = corrections.record_correction(root, event, 'wrong_label', 'bicycle')
            self.assertEqual(entry['label'], 'bicycle')
            self.assertEqual(corrections.read_correction(event)['verdict'], 'wrong_label')
            kept = list((root / 'corrections' / 'images').iterdir())
            self.assertEqual(len(kept), 2)            # image and its trigger crop survive retention
            self.assertEqual(len(corrections.load_corrections(root)), 1)
            with self.assertRaises(ValueError): corrections.record_correction(root, event, 'wrong_label', None)
            with self.assertRaises(ValueError): corrections.record_correction(root, event, 'nonsense')

    def test_per_home_package_is_preferred(self):
        import tempfile
        from pathlib import Path
        from detection.coreml_yolo import resolve_package
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'yolo11s.mlpackage').mkdir()
            self.assertEqual(resolve_package([folder], 's').name, 'yolo11s.mlpackage')
            (Path(folder) / 'yolo11s-home.mlpackage').mkdir()
            self.assertEqual(resolve_package([folder], 's').name, 'yolo11s-home.mlpackage')
            self.assertIsNone(resolve_package([folder], 'm'))
