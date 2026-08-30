"""Authentication for the native app's private loopback engine."""
import hmac
from http.cookies import SimpleCookie, CookieError
import os
import queue
import uuid

notifications = queue.Queue(maxsize=100)


def authorized(headers, port, token=None):
    token = os.environ.get('CLEARCAM_SESSION_TOKEN', '') if token is None else token
    if not token: return True  # Standalone developer mode only.
    expected_host = f'127.0.0.1:{port}'
    if headers.get('Host') != expected_host: return False
    origin = headers.get('Origin')
    if origin and origin != 'http://' + expected_host: return False
    credential = headers.get('Authorization', '').removeprefix('Bearer ')
    if not credential:
        try:
            cookie = SimpleCookie(headers.get('Cookie', ''))
            credential = cookie['ClearCamSession'].value if 'ClearCamSession' in cookie else ''
        except CookieError: return False
    return hmac.compare_digest(credential, token)


def enqueue_notification(title, body):
    try:
        notifications.put_nowait(dict(id=str(uuid.uuid4()), title=title, body=body))
        return True
    except queue.Full:
        return False


def take_notifications():
    result = []
    while True:
        try: result.append(notifications.get_nowait())
        except queue.Empty: return result
