import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils.native_session import authorized
from utils.runtime_paths import model_asset


class NativeSessionTests(unittest.TestCase):
    def test_requires_owned_host_and_secret(self):
        headers = {'Host': '127.0.0.1:5000'}
        self.assertFalse(authorized(headers, 5000, 'secret'))
        headers['Authorization'] = 'Bearer secret'
        self.assertTrue(authorized(headers, 5000, 'secret'))
        headers['Origin'] = 'https://attacker.invalid'
        self.assertFalse(authorized(headers, 5000, 'secret'))
        headers.pop('Origin')
        headers['Host'] = 'attacker.invalid:5000'
        self.assertFalse(authorized(headers, 5000, 'secret'))

    def test_cookie_authentication_for_video_and_webview(self):
        self.assertTrue(authorized({'Host': '127.0.0.1:5000', 'Cookie': 'ClearCamSession=secret'}, 5000, 'secret'))
        self.assertFalse(authorized({'Host': '127.0.0.1:5000', 'Cookie': 'ClearCamSession=wrong'}, 5000, 'secret'))

    def test_missing_bundled_model_fails_without_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, CLEARCAM_MODEL_DIR=folder):
            with self.assertRaisesRegex(RuntimeError, 'not included'):
                model_asset('https://example.invalid/model')
