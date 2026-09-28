import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import script.roboflow_model as rm


class PackagePathsTests(unittest.TestCase):
    def test_assist_path_is_fixed_name_independent_of_size(self):
        for size in ('t', 's', 'm'):
            target, previous, meta = rm._package_paths('/data', size, assist=True)
            self.assertEqual(target.name, 'vehicle-assist.mlpackage')
            self.assertEqual(previous.name, 'vehicle-assist.previous.mlpackage')
            self.assertEqual(meta.name, 'vehicle-assist.json')

    def test_non_assist_path_unchanged_by_size(self):
        target, previous, meta = rm._package_paths('/data', 's', assist=False)
        self.assertEqual(target.name, 'yolo11s-home.mlpackage')
        self.assertEqual(previous.name, 'yolo11s-home.previous.mlpackage')
        self.assertEqual(meta.name, 'yolo11s-home.json')

        target_t, _, _ = rm._package_paths('/data', 't', assist=False)
        self.assertEqual(target_t.name, 'yolo11n-home.mlpackage')

    def test_assist_and_non_assist_paths_live_under_models_dir(self):
        target, _, _ = rm._package_paths('/data', 's', assist=True)
        self.assertEqual(target.parent, Path('/data/models'))


class RollbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _make_package(self, path):
        path.mkdir(parents=True)
        (path / 'marker.txt').write_text('x')

    def test_rollback_assist_restores_previous_and_leaves_home_slot_alone(self):
        models_dir = self.tmp / 'models'
        assist_target, assist_previous, _ = rm._package_paths(self.tmp, 's', assist=True)
        home_target, _, _ = rm._package_paths(self.tmp, 's', assist=False)
        self._make_package(assist_previous)
        self._make_package(assist_target)
        (assist_target / 'marker.txt').write_text('current')
        self._make_package(home_target)

        rm.rollback(self.tmp, 's', assist=True)

        self.assertTrue(assist_target.exists())
        self.assertEqual((assist_target / 'marker.txt').read_text(), 'x')
        self.assertFalse(assist_previous.exists())
        # the size-based -home slot is a completely separate file/name and is
        # untouched by an --assist rollback.
        self.assertTrue(home_target.exists())

    def test_rollback_assist_without_previous_removes_target(self):
        assist_target, _, assist_meta = rm._package_paths(self.tmp, 's', assist=True)
        self._make_package(assist_target)
        assist_meta.write_text('{}')

        rm.rollback(self.tmp, 's', assist=True)

        self.assertFalse(assist_target.exists())
        self.assertFalse(assist_meta.exists())


class FromLocalInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_from_local_copies_package_to_assist_slot(self):
        source = self.tmp / 'weights.mlpackage'
        source.mkdir()
        (source / 'model.mlmodel').write_text('fake')

        argv = ['roboflow_model.py', '--assist', '--from-local', str(source), '--data', str(self.tmp)]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rm.main()
        finally:
            sys.argv = old_argv

        target, _, meta = rm._package_paths(self.tmp, 's', assist=True)
        self.assertTrue(target.exists())
        self.assertTrue((target / 'model.mlmodel').exists())
        self.assertTrue(meta.exists())

    def test_from_local_keeps_previous_on_reinstall(self):
        source = self.tmp / 'weights.mlpackage'
        source.mkdir()
        (source / 'model.mlmodel').write_text('fake')
        target, previous, _ = rm._package_paths(self.tmp, 's', assist=True)
        target.parent.mkdir(parents=True)
        target.mkdir()
        (target / 'old.txt').write_text('old')

        argv = ['roboflow_model.py', '--assist', '--from-local', str(source), '--data', str(self.tmp)]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rm.main()
        finally:
            sys.argv = old_argv

        self.assertTrue(previous.exists())
        self.assertTrue((previous / 'old.txt').exists())
        self.assertTrue((target / 'model.mlmodel').exists())


if __name__ == '__main__':
    unittest.main()
