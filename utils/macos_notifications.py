"""Small local-only macOS notification adapter."""

import subprocess
import os


def send(title, body=None):
    message = body or "A camera event was detected."
    if os.environ.get('CLEARCAM_NATIVE') == '1':
        from utils.native_session import enqueue_notification
        return enqueue_notification(title, message)
    # Pass text as arguments, not AppleScript source (including Unicode/quotes).
    script = 'on run argv\n display notification (item 2 of argv) with title (item 1 of argv)\nend run'
    result = subprocess.run(["osascript", "-e", script, title, message], check=False, capture_output=True, text=True)
    return result.returncode == 0
