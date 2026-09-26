"""Regression coverage without opening a camera or loading inference models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
import threading
from unittest.mock import Mock


class CaptureCommandTests(unittest.TestCase):
    def test_live_decoder_uses_supported_output_options(self):
        source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'VideoCapture')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_open_ffmpeg_locked')
        popen = Mock(return_value=object())
        namespace = {
            'uuid': uuid,
            'find_ffmpeg': lambda: '/test/ffmpeg',
            'subprocess': SimpleNamespace(Popen=popen, DEVNULL=-3, PIPE=-1),
            'time': SimpleNamespace(sleep=Mock(), time=lambda: 100),
            'DETECT_FPS': 10.0,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<capture>', 'exec'), namespace)
        camera = SimpleNamespace(
            _get_new_stream_dir=lambda name: Path('/test/streams'),
            proc={}, hls_proc={}, src={'test': 'rtsp://example.invalid/stream1'},
            vod={'test': False}, width={'test': 1920}, height={'test': 1080},
            start_time={'test': None},
        )
        namespace['_open_ffmpeg_locked'](camera, 'test')
        recorder, decoder = [call.args[0] for call in popen.call_args_list]
        self.assertIn('-rtsp_transport', recorder)
        self.assertIn('program_date_time', recorder[recorder.index('-hls_flags') + 1])
        self.assertNotEqual(recorder[recorder.index('-hls_segment_filename') + 1], '/test/streams/stream_%06d.ts')
        self.assertNotIn('-vsync', decoder)
        # Full camera frame rate reaches the live view; detection throttles itself in process_frame.
        self.assertNotIn('-fps_mode', decoder)
        self.assertEqual(decoder[decoder.index('-vf') + 1], 'scale=1920:1080')
        self.assertEqual(decoder[decoder.index('-pix_fmt') + 1], 'bgr24')
        self.assertNotIn('-reconnect', decoder)  # local HLS, not an HTTP input

    def test_short_read_does_not_reach_reshape(self):
        source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'VideoCapture')
        loop = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'frame_loop')
        short_read = next(n for n in ast.walk(loop) if isinstance(n, ast.If) and 'len(raw_bytes)' in ast.unparse(n.test))
        self.assertIsInstance(short_read.body[-1], ast.Continue)

    def test_shutdown_prevents_new_recorder(self):
        source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'VideoCapture')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_open_ffmpeg')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<restart>', 'exec'), namespace)
        camera = SimpleNamespace(restart_lock=threading.RLock(), stopping=threading.Event(),
                                 hls_proc={}, proc={}, _open_ffmpeg_locked=Mock(return_value=('recorder', 'decoder')))
        self.assertEqual(namespace['_open_ffmpeg'](camera, 'test'), ('recorder', 'decoder'))
        self.assertEqual(camera.hls_proc['test'], 'recorder')
        camera.stopping.set()
        namespace['_open_ffmpeg'](camera, 'test')
        camera._open_ffmpeg_locked.assert_called_once()


if __name__ == '__main__':
    unittest.main()


def _watchdog_methods(clock):
    source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'VideoCapture')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('decoder_stalled', 'kick_stalled_decoder')]
    namespace = {'time': SimpleNamespace(time=clock)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<watchdog>', 'exec'), namespace)
    return namespace


class DecoderWatchdogTests(unittest.TestCase):
    def camera(self, **state):
        alive = SimpleNamespace(poll=lambda: None)
        return SimpleNamespace(proc={'cam': alive}, pipeline={'cam': {**dict(last_frame=None, stream_started=None), **state}},
                               _safe_kill_process=Mock())

    def test_hang_before_first_frame_is_restarted(self):
        ns = _watchdog_methods(lambda: 200)
        cam = self.camera(stream_started=100)
        cam.decoder_stalled = lambda name, now=None, limit=30: ns['decoder_stalled'](cam, name, now, limit)
        self.assertTrue(ns['kick_stalled_decoder'](cam, 'cam'))
        cam._safe_kill_process.assert_called_once()

    def test_fresh_stream_is_left_alone(self):
        ns = _watchdog_methods(lambda: 110)
        cam = self.camera(stream_started=100)
        self.assertFalse(ns['decoder_stalled'](cam, 'cam', 110))

    def test_recent_frame_wins_over_old_stream_start(self):
        ns = _watchdog_methods(lambda: 200)
        cam = self.camera(stream_started=100, last_frame=190)
        self.assertFalse(ns['decoder_stalled'](cam, 'cam', 200))

    def test_kicks_are_rate_limited(self):
        ns = _watchdog_methods(lambda: 200)
        cam = self.camera(stream_started=100)
        cam.last_decoder_kick = {'cam': 185}
        self.assertFalse(ns['decoder_stalled'](cam, 'cam', 200))

    def test_dead_decoder_is_left_to_frame_loop(self):
        ns = _watchdog_methods(lambda: 200)
        cam = self.camera(stream_started=100)
        cam.proc['cam'] = SimpleNamespace(poll=lambda: 1)
        self.assertFalse(ns['decoder_stalled'](cam, 'cam', 200))
