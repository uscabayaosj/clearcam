"""Unit coverage for per-camera ignore areas: validation, the geometry that
drops detections before the tracker sees them, and the /edit_settings
handler that persists and validates them."""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils import ignore_areas


CLASS_NAMES = ['person', 'bicycle', 'car', 'motorbike', 'bus']


class ParseAreasTests(unittest.TestCase):
    def test_none_and_empty_are_a_noop(self):
        self.assertEqual(ignore_areas.parse_areas(None), [])
        self.assertEqual(ignore_areas.parse_areas([]), [])

    def test_valid_area_with_class_restriction(self):
        parsed = ignore_areas.parse_areas(
            [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}], CLASS_NAMES)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["box"], [0.86, 0.65, 1.0, 1.0])
        self.assertEqual(parsed[0]["classes"], {"car"})

    def test_missing_or_empty_classes_means_all_classes(self):
        parsed = ignore_areas.parse_areas([{"box": [0.0, 0.0, 0.5, 0.5]}], CLASS_NAMES)
        self.assertIsNone(parsed[0]["classes"])
        parsed = ignore_areas.parse_areas(
            [{"box": [0.0, 0.0, 0.5, 0.5], "classes": []}], CLASS_NAMES)
        self.assertIsNone(parsed[0]["classes"])

    def test_class_names_are_case_insensitive(self):
        parsed = ignore_areas.parse_areas(
            [{"box": [0.0, 0.0, 0.5, 0.5], "classes": ["CAR"]}], CLASS_NAMES)
        self.assertEqual(parsed[0]["classes"], {"car"})

    def test_rejects_not_a_list(self):
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas({"box": [0, 0, 1, 1]}, CLASS_NAMES)

    def test_rejects_too_many_areas(self):
        areas = [{"box": [0, 0, 0.01, 0.01]} for _ in range(ignore_areas.MAX_AREAS + 1)]
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas(areas, CLASS_NAMES)

    def test_rejects_bad_box_shape(self):
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas([{"box": [0, 0, 1]}], CLASS_NAMES)

    def test_rejects_non_numeric_box(self):
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas([{"box": ["a", 0, 1, 1]}], CLASS_NAMES)

    def test_rejects_out_of_range_or_inverted_coords(self):
        for box in ([-0.1, 0, 1, 1], [0, 0, 1.1, 1], [0.5, 0, 0.4, 1], [0, 0.5, 1, 0.4], [0, 0, 0, 1]):
            with self.assertRaises(ValueError):
                ignore_areas.parse_areas([{"box": box}], CLASS_NAMES)

    def test_rejects_unknown_class_name(self):
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas([{"box": [0, 0, 1, 1], "classes": ["bin"]}], CLASS_NAMES)

    def test_skips_class_validation_when_no_vocabulary_given(self):
        parsed = ignore_areas.parse_areas([{"box": [0, 0, 1, 1], "classes": ["bin"]}])
        self.assertEqual(parsed[0]["classes"], {"bin"})

    def test_rejects_non_string_or_blank_class_name(self):
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas([{"box": [0, 0, 1, 1], "classes": [""]}], CLASS_NAMES)
        with self.assertRaises(ValueError):
            ignore_areas.parse_areas([{"box": [0, 0, 1, 1], "classes": [5]}], CLASS_NAMES)


def pred(x1, y1, x2, y2, score, cls):
    return [x1, y1, x2, y2, score, cls]


class FilterPredsTests(unittest.TestCase):
    """1280x720 frame; the tapoc113 bins sit in the ignore box
    [0.86, 0.65, 1.0, 1.0], restricted to 'car' (the reported bug: bins
    called 'car' in ~40% of frames)."""
    WIDTH, HEIGHT = 1280, 720

    def test_empty_areas_is_a_noop(self):
        preds = np.array([pred(1100, 500, 1200, 650, 0.9, 2)])
        out = ignore_areas.filter_preds(preds, [], self.WIDTH, self.HEIGHT, CLASS_NAMES)
        np.testing.assert_array_equal(out, preds)

    def test_centre_inside_matching_class_is_dropped(self):
        areas = ignore_areas.parse_areas(
            [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}], CLASS_NAMES)
        # centre at (1152, 522) -> normalised (0.9, 0.725): inside the box.
        bin_as_car = pred(1100, 470, 1204, 574, 0.9, 2)
        out = ignore_areas.filter_preds(
            np.array([bin_as_car]), areas, self.WIDTH, self.HEIGHT, CLASS_NAMES)
        self.assertEqual(out.shape[0], 0)

    def test_centre_outside_the_box_is_kept(self):
        areas = ignore_areas.parse_areas(
            [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}], CLASS_NAMES)
        # A car in the middle of frame: centre well outside the ignore box.
        mid_frame_car = pred(500, 300, 600, 400, 0.9, 2)
        out = ignore_areas.filter_preds(
            np.array([mid_frame_car]), areas, self.WIDTH, self.HEIGHT, CLASS_NAMES)
        self.assertEqual(out.shape[0], 1)

    def test_class_restricted_area_keeps_other_classes(self):
        areas = ignore_areas.parse_areas(
            [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}], CLASS_NAMES)
        # A person, same spot as the bins: class doesn't match -> kept.
        person_same_spot = pred(1100, 470, 1204, 574, 0.9, 0)
        out = ignore_areas.filter_preds(
            np.array([person_same_spot]), areas, self.WIDTH, self.HEIGHT, CLASS_NAMES)
        self.assertEqual(out.shape[0], 1)

    def test_unrestricted_area_drops_every_class(self):
        areas = ignore_areas.parse_areas([{"box": [0.86, 0.65, 1.0, 1.0]}], CLASS_NAMES)
        for cls in (0, 2, 3):
            out = ignore_areas.filter_preds(
                np.array([pred(1100, 470, 1204, 574, 0.9, cls)]), areas, self.WIDTH, self.HEIGHT, CLASS_NAMES)
            self.assertEqual(out.shape[0], 0, f"class {cls} should have been dropped")

    def test_mixed_batch_drops_only_the_matching_row(self):
        areas = ignore_areas.parse_areas(
            [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}], CLASS_NAMES)
        bin_as_car = pred(1100, 470, 1204, 574, 0.9, 2)
        mid_frame_car = pred(500, 300, 600, 400, 0.9, 2)
        out = ignore_areas.filter_preds(
            np.array([bin_as_car, mid_frame_car]), areas, self.WIDTH, self.HEIGHT, CLASS_NAMES)
        self.assertEqual(out.shape[0], 1)
        np.testing.assert_array_equal(out[0], mid_frame_car)

    def test_empty_preds_is_a_noop(self):
        areas = ignore_areas.parse_areas([{"box": [0, 0, 1, 1]}], CLASS_NAMES)
        out = ignore_areas.filter_preds(np.empty((0, 6)), areas, self.WIDTH, self.HEIGHT, CLASS_NAMES)
        self.assertEqual(out.shape[0], 0)

    def test_none_class_names_only_matches_unrestricted_areas(self):
        restricted = ignore_areas.parse_areas(
            [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}])
        bin_as_car = pred(1100, 470, 1204, 574, 0.9, 2)
        out = ignore_areas.filter_preds(np.array([bin_as_car]), restricted, self.WIDTH, self.HEIGHT, None)
        self.assertEqual(out.shape[0], 1)  # can't resolve class name -> kept
        unrestricted = ignore_areas.parse_areas([{"box": [0.86, 0.65, 1.0, 1.0]}])
        out = ignore_areas.filter_preds(np.array([bin_as_car]), unrestricted, self.WIDTH, self.HEIGHT, None)
        self.assertEqual(out.shape[0], 0)  # no class check needed -> dropped


def _extract_edit_settings_handler():
    """Pulls just the `/edit_settings` branch out of clearcam.py's do_GET
    (a very long method) and wraps it as a standalone function, so this test
    exercises the real persisted-validation logic without booting a camera,
    a model, or an HTTP server."""
    source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
    do_get = next(
        n for n in ast.walk(source)
        if isinstance(n, ast.FunctionDef) and n.name == 'do_GET')
    branch = next(
        n for n in ast.walk(do_get)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "parsed_path.path == '/edit_settings'")
    func = ast.FunctionDef(
        name='_edit_settings',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in ('self', 'parsed_path', 'query', 'cam_name')],
                            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=branch.body,
        decorator_list=[], returns=None)
    ast.fix_missing_locations(func)
    namespace = {'json': json, 'ignore_areas_mod': ignore_areas, 'class_labels': CLASS_NAMES}
    exec(compile(ast.Module(body=[func], type_ignores=[]), '<edit_settings>', 'exec'), namespace)
    return namespace['_edit_settings']


class FakeSelf:
    def __init__(self, stored_settings, links=None):
        self._settings = dict(stored_settings)
        self._links = dict(links or {})
        self.responses = []
        self.refusal = None

    def _db(self):
        def run_get(table, key):
            # Every fake camera here is 'tapoc113'; a real `database.run_get`
            # would key on `key` too, but this stand-in only ever holds one
            # camera's settings, matching how the real handler scopes it.
            return dict(self._settings) if table == 'settings' else None

        def run_put(table, key, val, id=None):
            if table == 'settings':
                self._settings = val
            elif table == 'links':
                self._links[key] = val
        return SimpleNamespace(run_get=run_get, run_put=run_put)

    def send_refusal(self, message, code=400):
        self.refusal = (code, message)

    def send_response(self, code):
        self.responses.append(code)

    def send_header(self, *a): pass
    def end_headers(self): pass

    class _wfile:
        def write(self, *_a): pass
    wfile = _wfile()


class EditSettingsHandlerTests(unittest.TestCase):
    """Exercises the actual /edit_settings branch of clearcam.py: this is
    where ignore_areas gets validated and persisted next to the existing
    zone (`coords`/`outside`/`threshold`) settings."""

    def setUp(self):
        self.handler = _extract_edit_settings_handler()

    def _call(self, cam_settings, ignore_areas_param):
        self_obj = FakeSelf(cam_settings)
        database_ns = self_obj._db()
        parsed_path = SimpleNamespace(path='/edit_settings')
        query = {'ignore_areas': [ignore_areas_param]} if ignore_areas_param is not None else {}
        # The extracted branch closes over `database` as a free/global name.
        func = self.handler
        func.__globals__['database'] = database_ns
        func(self_obj, parsed_path, query, 'tapoc113')
        return self_obj

    def test_sets_ignore_areas_for_tapoc113_bins_as_cars(self):
        payload = json.dumps([{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}])
        result = self._call({}, payload)
        self.assertIsNone(result.refusal)
        self.assertEqual(result._settings['ignore_areas'], [{"box": [0.86, 0.65, 1.0, 1.0], "classes": ["car"]}])
        self.assertEqual(result.responses, [200])

    def test_invalid_payload_is_refused_and_not_persisted(self):
        payload = json.dumps([{"box": [0.86, 0.65, 1.0]}])  # bad box shape
        result = self._call({'threshold': 0.4}, payload)
        self.assertIsNotNone(result.refusal)
        self.assertEqual(result.refusal[0], 400)
        # settings untouched -- the DB write never happened
        self.assertEqual(result._settings, {'threshold': 0.4})

    def test_unknown_class_name_is_refused(self):
        payload = json.dumps([{"box": [0, 0, 1, 1], "classes": ["bin"]}])
        result = self._call({}, payload)
        self.assertIsNotNone(result.refusal)

    def test_empty_list_clears_previously_stored_areas(self):
        result = self._call({'ignore_areas': [{"box": [0, 0, 1, 1]}]}, json.dumps([]))
        self.assertIsNone(result.refusal)
        self.assertNotIn('ignore_areas', result._settings)

    def test_no_ignore_areas_param_leaves_settings_untouched(self):
        result = self._call({'threshold': 0.5}, None)
        self.assertIsNone(result.refusal)
        self.assertNotIn('ignore_areas', result._settings)
        self.assertEqual(result._settings['threshold'], 0.5)


if __name__ == '__main__':
    unittest.main()
