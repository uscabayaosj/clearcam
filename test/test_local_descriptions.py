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
