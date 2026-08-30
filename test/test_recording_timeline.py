import os
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from utils.recording_timeline import contained_path, read_timeline, position_at, event_timing, write_event_time, expired_recording_dirs


class RecordingTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_restart_gap_and_missing_segment(self):
        playlist = self.root / 'stream.m3u8'
        playlist.write_text('#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2026-08-30T12:00:00Z\n#EXTINF:2,\na.ts\n#EXT-X-DISCONTINUITY\n#EXT-X-PROGRAM-DATE-TIME:2026-08-30T12:05:00Z\n#EXTINF:2,\nb.ts\n')
        for name in ('a.ts', 'b.ts'): (self.root / name).touch()
        start = datetime.fromisoformat('2026-08-30T12:00:00+00:00').timestamp()
        os.utime(self.root / 'a.ts', (start + 2, start + 2))
        os.utime(self.root / 'b.ts', (start + 302, start + 302))
        timeline = read_timeline(playlist)
        self.assertEqual(position_at(timeline, start + 1), 1)
        self.assertIsNone(position_at(timeline, start + 60))
        self.assertEqual(position_at(timeline, start + 301), 3)
        (self.root / 'b.ts').unlink()
        self.assertIsNone(position_at(read_timeline(playlist), start + 301))

    def test_event_time_is_not_filename_or_engine_uptime(self):
        image = self.root / '900000000_notif.jpg'
        image.touch()
        write_event_time(image, 1000)
        timeline = [dict(offset=500, duration=10, wall=990, exists=True)]
        timing = event_timing(image, timeline)
        self.assertEqual(timing['captured_at'], 1000)
        self.assertEqual(timing['playback_offset'], 505)

    def test_appended_legacy_playlist_rejects_shifted_program_time(self):
        playlist = self.root / 'stream.m3u8'
        playlist.write_text('#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2026-08-30T12:00:00Z\n#EXTINF:2,\na.ts\n')
        segment = self.root / 'a.ts'
        segment.touch()
        os.utime(segment, (1002, 1002))
        self.assertEqual(position_at(read_timeline(playlist), 1001), 1)

    def test_legacy_time_and_unavailable_recording(self):
        image = self.root / '15_notif.jpg'
        image.touch()
        os.utime(image, (1000, 1000))
        timing = event_timing(image, [])
        self.assertEqual(timing['captured_at'], 1000)
        self.assertIsNone(timing['playback_offset'])

    def test_path_containment_including_symlinks(self):
        (self.root / 'link').symlink_to(self.root.parent)
        for path in ('../secret', '/etc/passwd', 'link/secret', '.'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                contained_path(self.root, path)
        self.assertEqual(contained_path(self.root, "Owner's camera/preview.png"), self.root / "Owner's camera/preview.png")

    def test_retention_never_selects_today_video_or_empty_camera(self):
        for folder in ('cam/streams/2026-08-29', 'cam/streams/2026-08-30', 'cam/streams/video', 'empty'):
            (self.root / folder).mkdir(parents=True)
        self.assertEqual(expired_recording_dirs(self.root, '2026-08-30'), [self.root / 'cam/streams/2026-08-29'])

    def test_live_playlist_serves_rolling_window_from_event_recording(self):
        from utils.recording_timeline import live_playlist
        playlist = self.root / 'stream.m3u8'
        entries = '#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-TARGETDURATION:4\n#EXT-X-PLAYLIST-TYPE:EVENT\n'
        for index in range(5):
            entries += f'#EXTINF:2,\nseg_{index}.ts\n'
        entries += '#EXT-X-DISCONTINUITY\n#EXTINF:2,\nseg_5.ts\n#EXTINF:2,\nseg_6.ts\n'
        playlist.write_text(entries)
        live = live_playlist(playlist)
        self.assertIn('#EXT-X-MEDIA-SEQUENCE:4', live)
        self.assertNotIn('#EXT-X-PLAYLIST-TYPE', live)
        self.assertNotIn('#EXT-X-ENDLIST', live)
        self.assertNotIn('seg_3.ts', live)
        for name in ('seg_4.ts', 'seg_5.ts', 'seg_6.ts'): self.assertIn(name, live)
        self.assertIn('#EXT-X-DISCONTINUITY\n#EXTINF:2,\nseg_5.ts', live)
        self.assertIn('#EXT-X-DISCONTINUITY-SEQUENCE:0', live)

    def test_live_playlist_window_opening_mid_session_drops_leading_discontinuity(self):
        from utils.recording_timeline import live_playlist
        playlist = self.root / 'stream.m3u8'
        playlist.write_text('#EXTM3U\n#EXT-X-TARGETDURATION:4\n'
                            '#EXTINF:2,\na.ts\n#EXT-X-DISCONTINUITY\n'
                            '#EXTINF:2,\nb.ts\n#EXTINF:2,\nc.ts\n#EXTINF:2,\nd.ts\n')
        live = live_playlist(playlist, window=3)
        self.assertIn('#EXT-X-DISCONTINUITY-SEQUENCE:1', live)
        self.assertNotIn('#EXT-X-DISCONTINUITY\n', live)
        self.assertIsNone(live_playlist(self.root / 'missing.m3u8'))
