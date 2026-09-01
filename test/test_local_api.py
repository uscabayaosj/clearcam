"""Exercise the real HTTP handler against isolated camera fixtures."""
import ast
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
import math
from utils import native_session
from datetime import datetime
from urllib.parse import urlparse, unquote, parse_qs, quote
from utils.recording_timeline import contained_path, read_timeline, event_timing, position_at


class LocalAPIRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for day in ('2026-08-30', 'video'):
            folder = self.root / 'cameras' / 'Test camera' / 'event_images' / day
            folder.mkdir(parents=True)
            (folder / '15_notif.jpg').write_bytes(b'fixture')
        (self.root / 'secret.db').write_text('not public')
        source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
        nodes = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ('HLSRequestHandler', 'image_sort_key')]
        from utils import household
        from utils import corrections
        namespace = dict(globals(), BASE_DIR=self.root,
                         global_settings=SimpleNamespace(use_face=False, use_clip=False),
                         read_description=lambda _: None, is_vod=lambda _: False,
                         household=household, corrections=corrections, household_store=household.HouseholdStore(self.root),
                         add_to_queue=lambda fn, *args: fn(*args),
                         enroll_household_face=lambda name, path: dict(error='No face was found in that image'))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<handler>', 'exec'), namespace)
        self.server = HTTPServer(('127.0.0.1', 0), namespace['HLSRequestHandler'])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, payload=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            conn.request('POST' if payload is not None else 'GET', path,
                         body=json.dumps(payload) if payload is not None else None,
                         headers={'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, response.read()
        finally: conn.close()

    def test_household_endpoints_validate_and_report(self):
        status, body = self.request('/household')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [])
        self.assertEqual(self.request('/household_enroll?name=Ana')[0], 400)
        self.assertEqual(self.request('/household_enroll?name=Ana&image=../secret.db')[0], 400)
        target = '/household_enroll?name=Ana&image=/Test%20camera/event_images/2026-08-30/15_notif.jpg'
        self.assertEqual(self.request(target)[0], 422)  # stub finds no face
        self.assertEqual(self.request('/household_delete?id=nope')[0], 400)
        self.assertEqual(self.request('/household_delete?id=' + '0' * 32)[0], 404)

    def test_media_traversal_and_camera_validation(self):
        self.assertEqual(self.request('/../secret.db')[0], 403)
        self.assertEqual(self.request('/%2e%2e/secret.db')[0], 403)
        self.assertEqual(self.request('/delete_camera?cam_name=..')[0], 400)
        self.assertEqual(self.request('/set_max_storage?max=nan')[0], 400)

    def test_explicit_date_does_not_include_uploaded_video(self):
        status, body = self.request('/event_thumbs', {'folder': '2026-08-30', 'start': 0, 'count': 100})
        self.assertEqual(status, 200)
        images = json.loads(body)['images']
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]['folder'], '2026-08-30')
        self.assertIn('Test%20camera', images[0]['url'])

    def test_video_is_not_duplicated_and_invalid_requests_return_errors(self):
        status, body = self.request('/event_thumbs', {'folder': 'video'})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)['images']), 1)
        self.assertEqual(self.request('/event_thumbs', {'start': -1})[0], 400)
        self.assertEqual(self.request('/event_thumbs', {'is_face': True})[0], 400)
