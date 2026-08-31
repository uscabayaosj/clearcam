"""Read-only application resources and separately owned writable user data."""
import os
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('CLEARCAM_DATA_DIR', RESOURCE_DIR / 'data')).expanduser().resolve()


def _writable_model_copy(target):
    """tinygrad opens weights O_RDWR even to read them.

    A bundle launched from a read-only mount (App Translocation, a disk image,
    an app installed by an admin for another user) therefore cannot load a
    bundled model at all. Materialize one writable copy beside the other
    caches and reuse it; the copy is keyed by name and size so a rebuilt model
    replaces a stale one.
    """
    import shutil
    cache_root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'clearcam-models'
    cache_root.mkdir(parents=True, exist_ok=True)
    copy = cache_root / target.name
    if copy.is_file() and copy.stat().st_size == target.stat().st_size:
        return copy
    staging = copy.with_suffix('.partial')
    shutil.copyfile(target, staging)
    staging.replace(copy)
    return copy


def model_asset(url):
    """Packaged builds never download a model implicitly."""
    bundled = os.environ.get('CLEARCAM_MODEL_DIR')
    if bundled:
        import hashlib
        target = Path(bundled) / hashlib.md5(url.encode()).hexdigest()
        if not target.is_file():
            raise RuntimeError('This model is not included in this ClearCam build. Use the bundled YOLO tiny and Qwen 2B models.')
        if not os.access(target, os.W_OK):
            return _writable_model_copy(target)
        return target
    from tinygrad.helpers import fetch
    return fetch(url)
