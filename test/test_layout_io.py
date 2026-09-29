"""Layout export/import between Macs: shape, validation, dry run, apply, handlers."""
import ast
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from utils import layout_io

CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog', 'stroller']
SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
ZONE = {"id": "z1", "name": "Yard", "polygon": SQUARE, "classes": ["person"], "metric": "passes"}
AREA = {"box": [0.6, 0.6, 0.9, 0.9], "classes": ["car"]}
ALERT = {"coords": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "outside": True}

STORED = {
    "coords": ALERT["coords"], "outside": True, "is_notif": True, "threshold": 0.42,
    "count_zones": [ZONE], "ignore_areas": [AREA], "url": "rtsp://user:secret@10.0.0.5/stream1", "reset": 1,
}


def layout_for(**cams):
    return {"clearcam_layout": 1, "exported_at": "2026-09-29T10:00:00+08:00", "cameras": cams}


ENTRY = {"count_zones": [ZONE], "ignore_areas": [AREA], "alert_zone": ALERT}


class BuildLayoutTests(unittest.TestCase):
    def test_export_shape_excludes_everything_else(self):
        layout = layout_io.build_layout({"tapoc113": STORED, "Empty": {}}, class_names=CLASS_NAMES)
        self.assertEqual(layout["clearcam_layout"], 1)
        datetime.fromisoformat(layout["exported_at"])
        self.assertEqual(set(layout["cameras"]), {"tapoc113", "Empty"})
        self.assertEqual(set(layout["cameras"]["tapoc113"]), {"count_zones", "ignore_areas", "alert_zone"})
        text = json.dumps(layout)
        for leaked in ("secret", "rtsp", "threshold", "is_notif", "0.42"):
            self.assertNotIn(leaked, text)
        self.assertEqual(layout["cameras"]["tapoc113"]["alert_zone"], ALERT)
        self.assertEqual(layout["cameras"]["Empty"], {"count_zones": [], "ignore_areas": [], "alert_zone": None})

    def test_round_trip(self):
        layout = json.loads(json.dumps(layout_io.build_layout({"a": STORED}, class_names=CLASS_NAMES)))
        parsed = layout_io.parse_layout(layout, CLASS_NAMES)
        self.assertEqual(parsed["a"]["count_zones"], [ZONE])
        self.assertEqual(parsed["a"]["ignore_areas"], [AREA])
        self.assertEqual(parsed["a"]["alert_zone"], ALERT)


class PlanAndApplyTests(unittest.TestCase):
    def test_dry_run_summary_with_unknown_camera(self):
        parsed = layout_io.parse_layout(layout_for(a=ENTRY, gone={"count_zones": [], "ignore_areas": [], "alert_zone": None}), CLASS_NAMES)
        plan = layout_io.plan_import(parsed, ["a", "b"])
        self.assertEqual(plan["apply"], ["a"])
        self.assertEqual(plan["skip"], ["gone"])
        self.assertEqual(plan["counts"], {"a": {"count_zones": 1, "ignore_areas": 1, "alert_zone": True}})

    def test_apply_leaves_unrelated_keys_intact(self):
        parsed = layout_io.parse_layout(layout_for(a={"count_zones": [], "ignore_areas": [AREA], "alert_zone": None}), CLASS_NAMES)
        out = layout_io.apply_entry(STORED, parsed["a"])
        self.assertNotIn("count_zones", out)
        self.assertNotIn("coords", out)
        self.assertEqual(out["ignore_areas"], [AREA])
        for key in ("is_notif", "threshold", "url", "reset"):
            self.assertEqual(out[key], STORED[key])
        self.assertEqual(STORED["coords"], ALERT["coords"])  # input not mutated

    def test_apply_sets_alert_zone(self):
        out = layout_io.apply_entry({"threshold": 0.3}, layout_io.parse_layout(layout_for(a=ENTRY), CLASS_NAMES)["a"])
        self.assertEqual(out["coords"], ALERT["coords"])
        self.assertIs(out["outside"], True)
        self.assertEqual(out["threshold"], 0.3)


class ValidationTests(unittest.TestCase):
    def test_rejects_whole_import_on_one_invalid_zone(self):
        bad = {**ENTRY, "count_zones": [{**ZONE, "metric": "bogus"}]}
        with self.assertRaises(ValueError) as ctx:
            layout_io.parse_layout(layout_for(good=ENTRY, broken=bad), CLASS_NAMES)
        self.assertIn("broken", str(ctx.exception))

    def test_rejects_bad_ignore_area_and_unknown_class(self):
        for entry in ({**ENTRY, "ignore_areas": [{"box": [0.5, 0.5, 0.2, 0.9]}]},
                      {**ENTRY, "ignore_areas": [{"box": [0, 0, 1, 1], "classes": ["unicorn"]}]}):
            with self.assertRaises(ValueError):
                layout_io.parse_layout(layout_for(a=entry), CLASS_NAMES)

    def test_alert_zone_validation(self):
        ok = layout_io.parse_alert_zone
        self.assertIsNone(ok(None))
        self.assertEqual(ok({"coords": SQUARE})["outside"], False)
        many = [[i / 60, 0.5] for i in range(50)]
        self.assertEqual(len(ok({"coords": many})["coords"]), 50)
        for bad in ({"coords": [[0, 0], [1, 1]]},
                    {"coords": many + [[0.5, 0.5]]},
                    {"coords": [[0, 0], [1, 1], [1.2, 0.5]]},
                    {"coords": [[0, 0], [1, 1], [-0.1, 0.5]]},
                    {"coords": [[0, 0], [1, 1], ["a", 0.5]]},
                    {"coords": [[0, 0], [1, 1], [0.5]]},
                    {"coords": SQUARE, "outside": "yes"},
                    {"outside": True}, [], "zone"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ok(bad)

    def test_rejects_wrong_file(self):
        for bad in ({}, {"clearcam_layout": 2, "cameras": {}}, {"clearcam_layout": 1}, [], {"clearcam_layout": 1, "cameras": []}):
            with self.assertRaises(ValueError):
                layout_io.parse_layout(bad, CLASS_NAMES)


class ExportFileTests(unittest.TestCase):
    def test_never_overwrites(self):
        now = datetime(2026, 9, 29, 12, 0)
        with tempfile.TemporaryDirectory() as d:
            names = [layout_io.write_export(d, {"x": i}, now).name for i in range(3)]
        self.assertEqual(names, ["ClearCam zones 2026-09-29.json", "ClearCam zones 2026-09-29 2.json", "ClearCam zones 2026-09-29 3.json"])


def _extract(func_name, test_source, arg_names):
    source = ast.parse((Path(__file__).parents[1] / 'clearcam.py').read_text())
    parent = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == func_name)
    branch = next(n for n in ast.walk(parent) if isinstance(n, ast.If) and ast.unparse(n.test) == test_source)
    func = ast.FunctionDef(
        name='_h', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in arg_names],
                                      kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=branch.body, decorator_list=[], returns=None)
    ast.fix_missing_locations(func)
    ns = {'json': json, 'layout_io': layout_io, 'class_labels': CLASS_NAMES,
          'parse_qs': __import__('urllib.parse', fromlist=['x']).parse_qs, 'Path': Path,
          'subprocess': SimpleNamespace(run=lambda *a, **k: None)}
    exec(compile(ast.Module(body=[func], type_ignores=[]), '<h>', 'exec'), ns)
    return ns['_h'], ns


class FakeServer:
    def __init__(self, links, settings, body=b''):
        self.links, self.settings = dict(links), {k: dict(v) for k, v in settings.items()}
        self.rfile = io.BytesIO(body)
        self.headers = {'Content-Length': str(len(body))}
        self.sent = None
        self.refusal = None
        self.db = SimpleNamespace(
            run_get=lambda table, key=None: self.links if table == 'links' else self.settings.get(key),
            run_put=lambda table, key, val, id=None: self.settings.__setitem__(key, val))

    def send_200(self, body=None): self.sent = body
    def send_refusal(self, message, code=400): self.refusal = (code, message)


class ImportHandlerTests(unittest.TestCase):
    def setUp(self):
        self.h, self.ns = _extract('do_POST', "parsed_path.path == '/import_layout'", ['self', 'parsed_path'])

    def _post(self, doc, query='', settings=None):
        body = json.dumps(doc).encode()
        srv = FakeServer({"a": "rtsp://x", "b": "rtsp://y"}, settings or {"a": dict(STORED), "b": {"threshold": 0.1}}, body)
        self.ns['database'] = srv.db
        self.h(srv, urlparse('/import_layout' + query))
        return srv

    def test_dry_run_changes_nothing(self):
        srv = self._post(layout_for(a=ENTRY, gone=ENTRY), '?dry_run=1')
        self.assertEqual(srv.sent["apply"], ["a"])
        self.assertEqual(srv.sent["skip"], ["gone"])
        self.assertEqual(srv.settings["a"], STORED)

    def test_invalid_part_rejects_everything(self):
        clear = {"count_zones": [], "ignore_areas": [], "alert_zone": None}
        srv = self._post(layout_for(a=clear, b={**ENTRY, "alert_zone": {"coords": [[0, 0]]}}))
        self.assertIsNotNone(srv.refusal)
        self.assertIn("Nothing was imported", srv.refusal[1])
        self.assertEqual(srv.settings["a"], STORED)

    def test_real_import_replaces_only_layout(self):
        srv = self._post(layout_for(a={"count_zones": [], "ignore_areas": [], "alert_zone": None}, gone=ENTRY))
        self.assertEqual(srv.sent["apply"], ["a"])
        self.assertEqual(srv.settings["a"], {"is_notif": True, "threshold": 0.42, "url": STORED["url"], "reset": 1})
        self.assertEqual(srv.settings["b"], {"threshold": 0.1})
        self.assertNotIn("gone", srv.settings)


class ExportHandlerTests(unittest.TestCase):
    def test_export_writes_file_and_returns_names(self):
        h, ns = _extract('do_GET', "parsed_path.path == '/export_layout'", ['self', 'parsed_path'])
        with tempfile.TemporaryDirectory() as home:
            ns['Path'] = SimpleNamespace(home=lambda: Path(home))
            srv = FakeServer({"a": "rtsp://user:pw@x", "blank": ""}, {"a": dict(STORED)})
            ns['database'] = srv.db
            h(srv, urlparse('/export_layout'))
            self.assertEqual(srv.sent["cameras"], ["a"])
            written = json.loads(Path(srv.sent["path"]).read_text())
            self.assertEqual(Path(srv.sent["path"]).parent, Path(home) / "Downloads")
            self.assertNotIn("pw", json.dumps(written))
            self.assertEqual(list(written["cameras"]), ["a"])


if __name__ == '__main__':
    unittest.main()
