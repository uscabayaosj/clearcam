"""Stage a relocatable local alpha, without camera data or developer paths.

Uses the installed python-build-standalone runtime as a build input, not as a
runtime dependency. Distribution licence/signing review remains a release gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
URLS = [
    'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
    'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-t.safetensors',
    'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf',
    'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf',
    'https://huggingface.co/roryclear/AdaFace/resolve/main/adaface_ir50_ms1mv2.safetensors',
]


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def signing_identity():
    """A stable local identity keeps TCC grants (Local Network) across rebuilds.

    Prefers the self-signed 'ClearCam Local Signing' certificate when the user
    has trusted it; otherwise falls back to ad-hoc signing, which macOS treats
    as a new app on every rebuild. CLEARCAM_SIGN_IDENTITY overrides.
    """
    if override := os.environ.get('CLEARCAM_SIGN_IDENTITY'):
        return override
    try:
        identities = run('/usr/bin/security', 'find-identity', '-v', '-p', 'codesigning')
    except subprocess.CalledProcessError:
        return '-'
    return 'ClearCam Local Signing' if 'ClearCam Local Signing' in identities else '-'


def copy_tree(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))


def is_macho(path):
    if path.is_symlink() or not path.is_file(): return False
    with path.open('rb') as stream: magic = stream.read(4)
    return magic in (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca')


def relocate_libraries(resources):
    """Bundle external dylibs and rewrite loads relative to each packaged file."""
    library_dir = resources / 'Libraries'
    library_dir.mkdir()
    pending = [p for p in resources.rglob('*') if is_macho(p)]
    visited, origins = set(), {}
    while pending:
        binary = pending.pop()
        if binary in visited: continue
        visited.add(binary)
        dependencies = run('/usr/bin/otool', '-L', str(binary)).splitlines()[1:]
        identities = run('/usr/bin/otool', '-D', str(binary)).splitlines()[1:]
        changes = []
        for entry in dependencies:
            dependency = entry.strip().split(' (compatibility')[0]
            if dependency in identities: continue
            if not dependency.startswith('/') or dependency.startswith(('/usr/lib/', '/System/Library/')): continue
            original = Path(dependency).resolve()
            destination = library_dir / original.name
            if destination.name in origins and origins[destination.name] != original:
                raise RuntimeError(f'Conflicting library names: {destination.name}')
            if not destination.exists():
                if not original.is_file(): raise RuntimeError(f'Missing runtime library: {original}')
                shutil.copy2(original, destination)
                destination.chmod(0o755)
                origins[destination.name] = original
                pending.append(destination)
            relative = '@loader_path/' + os.path.relpath(destination, binary.parent)
            changes.extend(['-change', dependency, relative])
        if changes:
            binary.chmod(binary.stat().st_mode | 0o200)
            run('/usr/bin/install_name_tool', *changes, str(binary))
        if binary.suffix == '.dylib':
            run('/usr/bin/install_name_tool', '-id', '@rpath/' + binary.name, str(binary))
        run('/usr/bin/codesign', '--force', '--sign', SIGN_IDENTITY, str(binary))
    return sorted(p.name for p in visited)


SIGN_IDENTITY = signing_identity()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()
    print('Signing identity:', 'ad-hoc (permissions reset each rebuild)' if SIGN_IDENTITY == '-' else SIGN_IDENTITY)
    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    # Stage outside dist: iCloud's fileproviderd re-stamps Finder metadata on
    # bundles inside synced folders (like ~/Documents), racing codesign forever.
    stage = Path(tempfile.mkdtemp(prefix='ClearCam-build-'))
    app = stage / 'ClearCam.app'
    contents = app / 'Contents'
    resources = contents / 'Resources'
    (contents / 'MacOS').mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(args.binary, contents / 'MacOS/ClearCam')
    plist = dict(CFBundleExecutable='ClearCam', CFBundleIdentifier='com.clearcam.mac.alpha',
                 CFBundleName='ClearCam', CFBundleDisplayName='ClearCam', CFBundlePackageType='APPL',
                 CFBundleShortVersionString='0.1.0', CFBundleVersion='1', LSMinimumSystemVersion='14.0',
                 NSPrincipalClass='NSApplication', NSHighResolutionCapable=True,
                 NSLocalNetworkUsageDescription='ClearCam connects to your home cameras to record and detect events on this Mac.',
                 NSAppTransportSecurity={'NSAllowsLocalNetworking': True, 'NSAllowsArbitraryLoadsInWebContent': True})
    (contents / 'Info.plist').write_bytes(plistlib.dumps(plist))
    engine = resources / 'Engine'
    engine.mkdir()
    for file in ['clearcam.py', 'mainview.html', 'LICENSE.md', 'requirements.txt']:
        shutil.copy2(ROOT / file, engine / file)
    for directory in ['utils', 'detection', 'models', 'ocsort_tracker', 'llm', 'vendor']:
        copy_tree(ROOT / directory, engine / directory)
    (engine / 'macos').mkdir()
    shutil.copy2(ROOT / 'macos/engine_bootstrap.py', engine / 'macos/engine_bootstrap.py')
    # Ship only runtime packages, never the whole developer virtual environment.
    runtime = resources / 'Runtime'
    copy_tree(Path(sys.base_prefix), runtime)
    target_site = runtime / 'lib/python3.11/site-packages'
    source_site = ROOT / '.venv/lib/python3.11/site-packages'
    for package in source_site.iterdir():
        if package.name.startswith(('numpy', 'cv2', 'opencv_python_headless', 'tinygrad',
                                    # Core ML detection runtime and its imports.
                                    'coremltools', 'PIL', 'pillow', 'google', 'protobuf',
                                    'sympy', 'mpmath', 'packaging', 'tqdm', 'attr', 'cattr',
                                    'pyaml', 'yaml', '_yaml', 'typing_extensions')):
            if package.is_dir(): copy_tree(package, target_site / package.name)
            else: shutil.copy2(package, target_site / package.name)
    # MLX description runtime: ~6x faster than tinygrad on Apple silicon. Copied
    # from a site-packages that has it installed (CLEARCAM_MLX_SITE); skipped,
    # with the tinygrad path remaining, when none is available.
    mlx_site = os.environ.get('CLEARCAM_MLX_SITE')
    mlx_bundled = False
    if mlx_site and Path(mlx_site).is_dir():
        for package in Path(mlx_site).iterdir():
            if package.name.startswith(('mlx', 'mlx_vlm', 'mlx_metal', 'transformers', 'tokenizers', 'huggingface_hub',
                                        'hf_xet', 'safetensors', 'jinja2', 'markupsafe', 'MarkupSafe', 'regex',
                                        'sentencepiece', 'requests', 'urllib3', 'charset_normalizer', 'idna',
                                        'certifi', 'filelock', 'fsspec', 'llguidance', 'scipy', 'miniaudio',
                                        'mlx_audio', 'fastapi', 'starlette', 'pydantic', 'pydantic_core',
                                        'annotated_types', 'typing_inspection', 'anyio', 'sniffio', 'websockets',
                                        'python_multipart', 'multipart', 'uvicorn', 'click', 'h11', 'httpx',
                                        'httpcore', 'shellingham', 'soundfile', '_soundfile', 'numba', 'llvmlite',
                                        'einops', 'audiofile', 'audresample', 'audmath', 'librosa', 'soxr',
                                        'joblib', 'threadpoolctl', 'scikit_learn', 'sklearn', 'decorator',
                                        'lazy_loader', 'msgpack', 'pooch', 'platformdirs', 'yarl', 'multidict',
                                        'aiohttp', 'aiosignal', 'frozenlist', 'attrs', 'propcache', 'aiohappyeyeballs')):
                if package.name.startswith(('numpy', 'cv2', 'opencv', 'PIL', 'pillow', 'tqdm', 'packaging')): continue
                if package.is_dir(): copy_tree(package, target_site / package.name)
                else: shutil.copy2(package, target_site / package.name)
        mlx_bundled = (target_site / 'mlx_vlm').is_dir() and (target_site / 'mlx').is_dir()
    # transformers verifies its dependencies through package *metadata*, so the
    # dist-info directories must travel with the packages or it refuses to import.
    if mlx_bundled:
        wanted = ('pyyaml', 'PyYAML', 'numpy', 'tokenizers', 'huggingface_hub', 'regex', 'requests', 'safetensors',
                  'packaging', 'filelock', 'tqdm', 'transformers', 'mlx', 'mlx_vlm', 'mlx_metal', 'jinja2',
                  'Jinja2', 'sentencepiece', 'pillow', 'Pillow', 'protobuf', 'typing_extensions', 'fsspec',
                  'hf_xet', 'urllib3', 'certifi', 'charset_normalizer', 'idna', 'markupsafe', 'MarkupSafe')
        for site in (source_site, Path(mlx_site)):
            for info in site.glob('*.dist-info'):
                if info.name.lower().startswith(tuple(w.lower() for w in wanted)) and not (target_site / info.name).exists():
                    copy_tree(info, target_site / info.name)
    tools = resources / 'Tools'
    tools.mkdir()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('FFmpeg is required on the build machine')
    shutil.copy2(ffmpeg, tools / 'ffmpeg')
    models = resources / 'Models'
    models.mkdir()
    from tinygrad.helpers import _ensure_downloads_dir
    inventory = []
    bundle_urls = list(URLS)
    if mlx_bundled:
        snapshot = None
        for root in (os.environ.get('CLEARCAM_MLX_MODELS'), Path.home() / '.cache/huggingface/hub'):
            if not root: continue
            base = Path(root) / 'models--mlx-community--Qwen3-VL-2B-Instruct-4bit' / 'snapshots'
            if base.is_dir():
                candidates = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
                if candidates: snapshot = candidates[0]; break
            if (Path(root) / 'Qwen3-VL-2B-Instruct-4bit' / 'config.json').is_file():
                snapshot = Path(root) / 'Qwen3-VL-2B-Instruct-4bit'; break
        if snapshot is None: raise RuntimeError('MLX runtime bundled but the Qwen3-VL-2B MLX snapshot is not in the HF cache')
        target = models / 'mlx' / 'Qwen3-VL-2B-Instruct-4bit'
        target.mkdir(parents=True)
        total = 0
        for item in snapshot.iterdir():
            real = item.resolve()
            shutil.copy2(real, target / item.name); total += real.stat().st_size
        inventory.append(dict(url='mlx-community/Qwen3-VL-2B-Instruct-4bit', file='mlx/Qwen3-VL-2B-Instruct-4bit', bytes=total))
        # The GGUF pair is the tinygrad fallback; with MLX aboard it would only double the download.
        bundle_urls = [u for u in URLS if 'Qwen3-VL' not in u]
    for url in bundle_urls:
        key = hashlib.md5(url.encode()).hexdigest()
        cached = _ensure_downloads_dir() / key
        if not cached.is_file(): raise RuntimeError(f'Required model must be prefetched on the build machine: {url}')
        shutil.copy2(cached, models / key)
        with cached.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        inventory.append(dict(url=url, file=key, bytes=cached.stat().st_size, sha256=checksum))
    binaries = relocate_libraries(resources)
    (resources / 'build-manifest.json').write_text(json.dumps(dict(models=inventory, binaries=binaries, architecture='arm64', channel='local-alpha', describer='mlx' if mlx_bundled else 'tinygrad'), indent=2))
    # Symlinks must remain within the app, including Python's command aliases.
    for link in app.rglob('*'):
        if link.is_symlink() and not link.resolve().is_relative_to(app.resolve()):
            raise RuntimeError(f'External symlink in bundle: {link}')
    # Finder metadata (copied in with sources, or stamped onto the bundle as
    # Finder notices it) makes codesign reject the bundle — strip before each step.
    def strip_finder_metadata():
        for attribute in ('com.apple.FinderInfo', 'com.apple.ResourceFork', 'com.apple.quarantine'):
            subprocess.run(['/usr/bin/xattr', '-rd', attribute, str(app)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for attempt in range(5):
        try:
            strip_finder_metadata()
            run('/usr/bin/codesign', '--force', '--deep', '--sign', SIGN_IDENTITY, str(app))
            strip_finder_metadata()
            run('/usr/bin/codesign', '--verify', '--deep', '--strict', str(app))
            break
        except subprocess.CalledProcessError:
            if attempt == 4: raise
    final = dist / 'ClearCam.app'
    if final.exists():
        previous = dist / ('ClearCam-previous-' + stage.name.removeprefix('ClearCam-build-') + '.app')
        final.rename(previous)
        print(f'Previous alpha preserved at {previous}')
    shutil.move(str(app), str(final))
    stage.rmdir()
    print(final)


if __name__ == '__main__': main()
