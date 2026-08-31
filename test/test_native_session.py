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

    def test_read_only_bundle_model_is_copied_somewhere_writable(self):
        """tinygrad opens weights O_RDWR; a translocated bundle is read-only."""
        import hashlib, stat
        url = 'https://example.invalid/weights.gguf'
        with tempfile.TemporaryDirectory() as models, tempfile.TemporaryDirectory() as cache:
            bundled = Path(models) / hashlib.md5(url.encode()).hexdigest()
            bundled.write_bytes(b'weights')
            os.chmod(bundled, stat.S_IRUSR)
            with patch.dict(os.environ, CLEARCAM_MODEL_DIR=models, XDG_CACHE_HOME=cache):
                usable = model_asset(url)
                self.assertNotEqual(usable, bundled)
                self.assertTrue(os.access(usable, os.W_OK))
                self.assertEqual(usable.read_bytes(), b'weights')
                self.assertEqual(model_asset(url), usable)  # reuses the copy

    def test_writable_bundle_model_is_used_in_place(self):
        import hashlib
        url = 'https://example.invalid/weights.gguf'
        with tempfile.TemporaryDirectory() as models:
            bundled = Path(models) / hashlib.md5(url.encode()).hexdigest()
            bundled.write_bytes(b'weights')
            with patch.dict(os.environ, CLEARCAM_MODEL_DIR=models):
                self.assertEqual(model_asset(url), bundled)
