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
        self.assertIn('-fps_mode', decoder)
        self.assertEqual(decoder[decoder.index('-fps_mode') + 1], 'vfr')
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
