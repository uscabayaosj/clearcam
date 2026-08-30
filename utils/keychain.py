"""macOS Keychain storage for local camera stream credentials."""

import subprocess
from urllib.parse import quote, unquote


SERVICE = "com.clearcam.camera-stream"
PREFIX = "keychain://"


def is_reference(value):
    return isinstance(value, str) and value.startswith(PREFIX)


def _account(reference_or_name):
    return unquote(reference_or_name[len(PREFIX):]) if is_reference(reference_or_name) else reference_or_name


def store(camera_name, source):
    """Store a source URL in the logged-in user's Keychain and return a safe reference."""
    subprocess.run(
        ["security", "add-generic-password", "-U", "-s", SERVICE, "-a", camera_name, "-w", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return PREFIX + quote(camera_name, safe="")


def retrieve(camera_name, stored_value):
    """Resolve a Keychain reference without ever returning the secret to HTTP clients."""
    if not is_reference(stored_value):
        return stored_value
    result = subprocess.run(
        ["security", "find-generic-password", "-s", SERVICE, "-a", _account(stored_value), "-w"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def remove(camera_name):
    subprocess.run(
        ["security", "delete-generic-password", "-s", SERVICE, "-a", camera_name],
        check=False,
        capture_output=True,
        text=True,
    )
