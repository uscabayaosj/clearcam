"""Keep camera credentials out of the engine log.

ffmpeg prints the input URL when it fails ("Error opening input file
rtsp://user:password@host/..."). Its stderr goes to the engine log, so every
line it writes passes through here first.
"""
import re
import subprocess
import sys
import threading

_CREDENTIALS = re.compile(r'(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@]*@')


def redact(text):
    """rtsp://user:pass@host -> rtsp://***@host (any scheme, any number of URLs)."""
    return _CREDENTIALS.sub(r'\1***@', text)


def _pump(stream, sink):
    try:
        for raw in iter(stream.readline, b''):
            sink.write(redact(raw.decode('utf-8', 'replace')))
            sink.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def popen_redacted(command, sink=None, **kwargs):
    """subprocess.Popen whose stderr reaches `sink` (default: our stderr) with
    credentials masked."""
    process = subprocess.Popen(command, stderr=subprocess.PIPE, **kwargs)
    threading.Thread(target=_pump, args=(process.stderr, sink or sys.stderr), daemon=True,
                     name='ffmpeg-stderr').start()
    return process
