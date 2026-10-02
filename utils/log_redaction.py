"""Keep camera credentials out of the engine log.

ffmpeg prints the input URL when it fails ("Error opening input file
rtsp://user:password@host/..."). Its stderr goes to the engine log, so every
line it writes passes through here first.
"""
import re
import subprocess
import sys
import threading

_URL = re.compile(r'(\b[a-zA-Z][a-zA-Z0-9+.-]*://)(\S+)')


def _mask(match):
    scheme, rest = match.group(1), match.group(2)
    # Passwords are typed in raw and can contain '/' or '@', so the login is
    # everything up to the LAST '@' in the URL (over-masking a rare '@' in a
    # path is fine; leaking a password is not).
    return scheme + '***@' + rest.rsplit('@', 1)[1] if '@' in rest else match.group(0)


def redact(text):
    """rtsp://user:pass@host -> rtsp://***@host (any scheme, any number of URLs)."""
    return _URL.sub(_mask, text)


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
