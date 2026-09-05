import importlib.util
import plistlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_packager():
    spec = importlib.util.spec_from_file_location('package_macos', ROOT / 'script/package_macos.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_entitlements_are_a_valid_plist_with_hardened_runtime_exceptions(self):
        entitlements = plistlib.loads((ROOT / 'macos/ClearCam.entitlements').read_bytes())
        for key in ('com.apple.security.cs.allow-jit', 'com.apple.security.cs.allow-unsigned-executable-memory',
                    'com.apple.security.cs.disable-library-validation'):
            self.assertIs(entitlements[key], True)
        # No sandbox: the engine must reach cameras on the LAN and write recordings freely.
        self.assertNotIn('com.apple.security.app-sandbox', entitlements)

    def test_executables_get_entitlements_but_libraries_do_not(self):
        packager = load_packager()
        self.assertTrue(packager.is_executable_macho(Path('/bin/ls')))
        extension = next(p for p in (ROOT / '.venv/lib').rglob('*.so') if packager.is_macho(p))
        self.assertFalse(packager.is_executable_macho(extension))
        self.assertIsInstance(packager.SIGN_TIMESTAMP, bool)


if __name__ == '__main__':
    unittest.main()
