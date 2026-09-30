import io
import sys
import time
import unittest
from utils.log_redaction import redact, popen_redacted


class RedactTest(unittest.TestCase):
  def test_rtsp_credentials_are_masked(self):
    self.assertEqual(redact('Error opening input file rtsp://cam:s3cret@192.168.1.2/stream1.'),
                     'Error opening input file rtsp://***@192.168.1.2/stream1.')

  def test_text_without_credentials_is_untouched(self):
    for line in ('[h264 @ 0x7c45] error while decoding MB 6 3', 'rtsp://192.168.1.2:554/live/ch0 failed'):
      self.assertEqual(redact(line), line)

  def test_every_url_on_a_line_is_masked(self):
    self.assertEqual(redact('a rtsp://u:p@h1/x b http://u2:p2@h2/y'), 'a rtsp://***@h1/x b http://***@h2/y')


class PopenRedactedTest(unittest.TestCase):
  def test_child_stderr_reaches_the_sink_masked(self):
    sink = io.StringIO()
    p = popen_redacted([sys.executable, '-c', "import sys; sys.stderr.write('open rtsp://a:b@h/s failed\\n')"], sink=sink)
    p.wait()
    for _ in range(50):
      if sink.getvalue(): break
      time.sleep(0.05)
    self.assertEqual(sink.getvalue(), 'open rtsp://***@h/s failed\n')


if __name__ == '__main__':
  unittest.main()
