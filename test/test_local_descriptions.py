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


class MlxDescriberTests(unittest.TestCase):
    def test_local_model_dir_and_sizes_follow_what_is_on_disk(self):
        import os, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from utils import mlx_describer
        with tempfile.TemporaryDirectory() as folder:
            two = Path(folder) / 'mlx' / 'Qwen3-VL-2B-Instruct-4bit'
            two.mkdir(parents=True); (two / 'config.json').write_text('{}')
            with patch.dict(os.environ, CLEARCAM_MODEL_DIR=folder, CLEARCAM_NATIVE='1'):
                self.assertEqual(mlx_describer.local_model_dir(2), two)
                self.assertIsNone(mlx_describer.local_model_dir(8))
                self.assertEqual(mlx_describer.available_sizes(), [2])
            with patch.dict(os.environ, {'CLEARCAM_NATIVE': '0'}):
                self.assertEqual(mlx_describer.available_sizes(), [2, 8])

    def test_factory_prefers_mlx_when_present_and_honours_override(self):
        import os
        from unittest.mock import patch
        from utils import local_descriptions, mlx_describer
        from utils.qwen_process import QwenProcess
        with patch.object(mlx_describer, 'runtime_available', return_value=True):
            self.assertIs(local_descriptions.default_model_factory(), mlx_describer.MlxProcess)
            with patch.dict(os.environ, CLEARCAM_DESCRIBER='tinygrad'):
                self.assertIs(local_descriptions.default_model_factory(), QwenProcess)
        with patch.object(mlx_describer, 'runtime_available', return_value=False):
            self.assertIs(local_descriptions.default_model_factory(), QwenProcess)


class MlxDownloadTests(unittest.TestCase):
    def test_download_refuses_unknown_sizes_and_reports_idle(self):
        import os, tempfile
        from unittest.mock import patch
        from utils import mlx_describer
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, CLEARCAM_DATA_DIR=folder):
            with self.assertRaises(ValueError): mlx_describer.start_download(3)
            self.assertEqual(mlx_describer.download_progress()['state'], 'idle')
            self.assertTrue(str(mlx_describer.download_dir(8)).endswith('models/mlx/Qwen3-VL-8B-Instruct-4bit'))


class TriggerPromptArrayTests(unittest.TestCase):
    def test_numpy_box_does_not_raise_and_locates_the_subject(self):
        import numpy as np
        from utils.local_descriptions import trigger_prompt
        box = np.array([1500.0, 800.0, 1800.0, 1000.0])
        prompt = trigger_prompt('car', box, 1920, 1080)
        self.assertIn('bottom right', prompt)
        self.assertTrue(prompt.startswith('A car was detected'))
        self.assertNotIn('of the frame', trigger_prompt('car', None, 1920, 1080))


class BoundedImageTests(unittest.TestCase):
    def test_large_frames_are_shrunk_and_small_ones_untouched(self):
        import os, tempfile
        from PIL import Image
        from utils.mlx_describer import bounded_image
        with tempfile.TemporaryDirectory() as d:
            big = os.path.join(d, 'big.jpg'); Image.new('RGB', (2304, 1296)).save(big)
            small = os.path.join(d, 'small.jpg'); Image.new('RGB', (900, 500)).save(small)
            out = bounded_image(big, 1024)
            self.assertNotEqual(out, big)
            with Image.open(out) as im: self.assertEqual(im.size, (1024, 576))
            os.unlink(out)
            self.assertEqual(bounded_image(small, 1024), small)
            self.assertEqual(bounded_image(big, 0), big)
