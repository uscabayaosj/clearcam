import io
import json
import os
import stat
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import roboflow_sync as rs

CLASS_NAMES = ['person', 'car', 'dog']


def make_entry(verdict, label=None, image='img.jpg'):
    return dict(
        time=1.0, verdict=verdict, label=label, image=image, crop=None, camera='cam1',
        width=100, height=50,
        detections=[
            dict(box=[1, 2, 3, 4], score=0.9, cls=0, label='person', trigger=True),
            dict(box=[5, 6, 7, 8], score=0.5, cls=1, label='car', trigger=False),
        ],
    )


class AnnotationXmlTests(unittest.TestCase):
    def test_confirm_keeps_all_labels(self):
        entry = make_entry('confirm')
        xml = rs.annotation_xml(entry, CLASS_NAMES)
        self.assertIn('<name>person</name>', xml)
        self.assertIn('<name>car</name>', xml)
        self.assertIn('<width>100</width>', xml)
        self.assertIn('<height>50</height>', xml)

    def test_wrong_label_replaces_trigger_only(self):
        entry = make_entry('wrong_label', label='dog')
        xml = rs.annotation_xml(entry, CLASS_NAMES)
        self.assertIn('<name>dog</name>', xml)
        self.assertIn('<name>car</name>', xml)
        self.assertNotIn('<name>person</name>', xml)

    def test_not_object_drops_trigger(self):
        entry = make_entry('not_object')
        xml = rs.annotation_xml(entry, CLASS_NAMES)
        self.assertNotIn('<name>person</name>', xml)
        self.assertIn('<name>car</name>', xml)

    def test_escapes_labels(self):
        entry = make_entry('wrong_label', label='a & b <c>')
        xml = rs.annotation_xml(entry, CLASS_NAMES)
        self.assertIn('a &amp; b &lt;c&gt;', xml)
        self.assertNotIn('a & b <c>', xml)


class ConfigTests(unittest.TestCase):
    def test_save_config_permissions_and_public_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = rs.save_config(tmp, dict(enabled=True, api_key='secret1234', project='p', workspace='w', bogus='x'))
            self.assertTrue(cfg['enabled'])
            self.assertEqual(cfg['api_key'], 'secret1234')
            self.assertNotIn('bogus', cfg)
            path = Path(tmp) / rs.CONFIG_FILE
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o600)

            pub = rs.public_config(cfg)
            self.assertNotIn('api_key', pub)
            self.assertTrue(pub['has_key'])
            self.assertEqual(pub['key_hint'], '1234')

    def test_load_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = rs.load_config(tmp)
            self.assertEqual(cfg, dict(enabled=False, workspace='', project='', api_key=''))


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._body


class FakeOpener:
    """Sequential fake responses; can raise HTTPError on a given call index."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.requests = []

    def __call__(self, request, timeout=30):
        self.calls.append(request.full_url)
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def write_correction_store(data_root, entries):
    store = Path(data_root) / 'corrections'
    images = store / 'images'
    images.mkdir(parents=True, exist_ok=True)
    with (store / 'corrections.jsonl').open('w') as f:
        for entry in entries:
            (images / entry['image']).write_bytes(b'jpg')
            f.write(json.dumps(entry) + '\n')


class SyncTests(unittest.TestCase):
    def test_requests_match_the_roboflow_sdk_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs.save_config(tmp, dict(enabled=True, api_key='key123', project='proj', workspace='ws'))
            write_correction_store(tmp, [make_entry('wrong_label', image='a.jpg', label='dog')])
            opener = FakeOpener([dict(success=True, id='id-a'), dict(success=True)])
            rs.sync(tmp, CLASS_NAMES, opener=opener)
            upload, annotate = opener.requests
            self.assertIn('/dataset/proj/upload?', upload.full_url)
            self.assertTrue(upload.get_header('Content-type').startswith('multipart/form-data; boundary='))
            self.assertIn(b'name="file"; filename="a.jpg"', upload.data)
            self.assertIn(b'name="split"\r\n\r\ntrain', upload.data)
            self.assertIn('/dataset/proj/annotate/id-a?', annotate.full_url)
            self.assertIn('overwrite=true', annotate.full_url)
            self.assertEqual(annotate.get_header('Content-type'), 'application/json')
            body = json.loads(annotate.data)
            self.assertIn('<name>dog</name>', body['annotationFile'])
            self.assertNotIn('key123', str(rs.status(tmp)))

    def test_sync_uploads_and_ledgers(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs.save_config(tmp, dict(enabled=True, api_key='key123', project='proj', workspace='ws'))
            entries = [make_entry('confirm', image='a.jpg'), make_entry('confirm', image='b.jpg')]
            write_correction_store(tmp, entries)

            opener = FakeOpener([
                dict(success=True, id='id-a'), dict(success=True),
                dict(success=True, id='id-b'), dict(success=True),
            ])
            result = rs.sync(tmp, CLASS_NAMES, opener=opener)
            self.assertEqual(result['uploaded_total'], 2)
            self.assertEqual(result['pending'], 0)
            uploaded = rs.uploaded_images(tmp)
            self.assertEqual(uploaded, {'a.jpg', 'b.jpg'})

    def test_sync_skips_already_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs.save_config(tmp, dict(enabled=True, api_key='key123', project='proj', workspace='ws'))
            entries = [make_entry('confirm', image='a.jpg')]
            write_correction_store(tmp, entries)

            opener1 = FakeOpener([dict(success=True, id='id-a'), dict(success=True)])
            rs.sync(tmp, CLASS_NAMES, opener=opener1)

            opener2 = FakeOpener([])  # should not be called
            result = rs.sync(tmp, CLASS_NAMES, opener=opener2)
            self.assertEqual(result['pending'], 0)
            self.assertEqual(opener2.calls, [])

    def test_sync_records_http_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs.save_config(tmp, dict(enabled=True, api_key='key123', project='proj', workspace='ws'))
            entries = [make_entry('confirm', image='a.jpg')]
            write_correction_store(tmp, entries)

            err = urllib.error.HTTPError('url', 500, 'boom', {}, io.BytesIO(b'server exploded'))
            opener = FakeOpener([err])
            result = rs.sync(tmp, CLASS_NAMES, opener=opener)
            self.assertIsNotNone(result['last_error'])
            self.assertIn('500', result['last_error'])
            self.assertEqual(result['uploaded_total'], 0)

    def test_sync_disabled_returns_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = [make_entry('confirm', image='a.jpg')]
            write_correction_store(tmp, entries)
            result = rs.sync(tmp, CLASS_NAMES)
            self.assertIsNotNone(result['last_error'])
            self.assertIn('not enabled', result['last_error'])
            self.assertEqual(result['uploaded_total'], 0)
            self.assertEqual(rs.uploaded_images(tmp), set())

    def test_sync_missing_key_returns_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs.save_config(tmp, dict(enabled=True, project='proj'))
            entries = [make_entry('confirm', image='a.jpg')]
            write_correction_store(tmp, entries)
            result = rs.sync(tmp, CLASS_NAMES)
            self.assertIn('api_key', result['last_error'])


class MergeAndRemapTests(unittest.TestCase):
    def test_merge_class_names(self):
        base = ['person', 'car']
        extra = ['car', 'bicycle', 'person', 'raccoon']
        merged = rs.merge_class_names(base, extra)
        self.assertEqual(merged, ['person', 'car', 'bicycle', 'raccoon'])

    def test_remap_yolo_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / 'labels'
            labels.mkdir()
            (labels / 'a.txt').write_text('0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.1 0.1\n')
            (labels / 'b.txt').write_text('2 0.5 0.5 0.2 0.2\n')
            out = Path(tmp) / 'out'
            index_map = {0: 10, 1: 11}  # class 2 unknown -> dropped
            count = rs.remap_yolo_labels(labels, out, index_map)
            self.assertEqual(count, 2)
            a_lines = (out / 'a.txt').read_text().splitlines()
            self.assertEqual(a_lines, ['10 0.1 0.1 0.2 0.2', '11 0.3 0.3 0.1 0.1'])
            b_lines = (out / 'b.txt').read_text().splitlines()
            self.assertEqual(b_lines, [])


class ReadYamlNamesTests(unittest.TestCase):
    def test_inline_list_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / 'data.yaml'
            yaml_path.write_text("train: train/images\nval: valid/images\nnames: ['cat', 'dog', 'bird']\n")
            result = rs.read_yaml_names(yaml_path)
            self.assertEqual(result['names'], ['cat', 'dog', 'bird'])
            self.assertEqual(result['train'], Path(tmp) / 'train/images')
            self.assertEqual(result['val'], Path(tmp) / 'valid/images')

    def test_block_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / 'data.yaml'
            yaml_path.write_text(
                "train: ../train/images\nval: ../valid/images\nnames:\n  0: cat\n  1: dog\n  2: bird\n"
            )
            result = rs.read_yaml_names(yaml_path)
            self.assertEqual(result['names'], ['cat', 'dog', 'bird'])


if __name__ == '__main__':
    unittest.main()
