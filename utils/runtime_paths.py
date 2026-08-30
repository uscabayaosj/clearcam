"""Read-only application resources and separately owned writable user data."""
import os
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('CLEARCAM_DATA_DIR', RESOURCE_DIR / 'data')).expanduser().resolve()


def model_asset(url):
    """Packaged builds never download a model implicitly."""
    bundled = os.environ.get('CLEARCAM_MODEL_DIR')
    if bundled:
        import hashlib
        target = Path(bundled) / hashlib.md5(url.encode()).hexdigest()
        if not target.is_file():
            raise RuntimeError('This model is not included in this ClearCam build. Use the bundled YOLO tiny and Qwen 2B models.')
        return target
    from tinygrad.helpers import fetch
    return fetch(url)
