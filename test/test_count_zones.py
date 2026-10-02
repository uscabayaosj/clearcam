"""Unit coverage for per-camera counting zones: validation, footpoint
point-in-polygon, the entry/exit hysteresis state machine, dwell/passes
aggregation, day rollover, save/load, and the /edit_settings handler."""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils import count_zones


CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog', 'stroller']

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def zone(id='z1', name='Yard', polygon=SQUARE, classes=('person',), metric='passes'):
    return {"id": id, "name": name, "polygon": [list(p) for p in polygon],
            "classes": list(classes), "metric": metric}


def make_counter(*args, **kwargs):
    # Existing tests start counting and feed objects at the same instant;
    # the startup grace (objects already present aren't new entries) has its own tests.
    kwargs.setdefault('startup_grace', 0.0)
    kwargs.setdefault('min_park', 0.0)   # short test stays; the parking minimum has its own tests
    return count_zones.ZoneCounter(*args, **kwargs)


class ParseZonesTests(unittest.TestCase):
    def test_none_and_empty_are_a_noop(self):
        self.assertEqual(count_zones.parse_zones(None), [])
        self.assertEqual(count_zones.parse_zones([]), [])

    def test_valid_zone_round_trips(self):
        parsed = count_zones.parse_zones([zone()], CLASS_NAMES)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "z1")
        self.assertEqual(parsed[0]["name"], "Yard")
        self.assertEqual(parsed[0]["classes"], ["person"])
        self.assertEqual(parsed[0]["metric"], "passes")
        self.assertEqual(len(parsed[0]["polygon"]), 4)

    def test_rejects_not_a_list(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones({"id": "z1"}, CLASS_NAMES)

    def test_rejects_too_many_zones(self):
        zones = [zone(id=f'z{i}') for i in range(count_zones.MAX_ZONES + 1)]
        with self.assertRaises(ValueError):
            count_zones.parse_zones(zones, CLASS_NAMES)

    def test_rejects_missing_or_blank_id(self):
        for bad in (None, '', '   ', 5):
            with self.assertRaises(ValueError):
                count_zones.parse_zones([{**zone(), "id": bad}], CLASS_NAMES)

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([zone(id='z1'), zone(id='z1')], CLASS_NAMES)

    def test_rejects_missing_or_blank_name(self):
        for bad in (None, '', '   '):
            with self.assertRaises(ValueError):
                count_zones.parse_zones([{**zone(), "name": bad}], CLASS_NAMES)

    def test_rejects_name_too_long(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "name": 'x' * (count_zones.MAX_NAME_LEN + 1)}], CLASS_NAMES)

    def test_rejects_too_few_or_too_many_polygon_points(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "polygon": [[0, 0], [1, 1]]}], CLASS_NAMES)
        too_many = [[0.01 * i, 0.01 * i] for i in range(count_zones.MAX_POLYGON_POINTS + 1)]
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "polygon": too_many}], CLASS_NAMES)

    def test_rejects_out_of_range_polygon_coords(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "polygon": [[-0.1, 0], [0.5, 0.5], [0.5, 1]]}], CLASS_NAMES)
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "polygon": [[0, 0], [1.5, 0.5], [0.5, 1]]}], CLASS_NAMES)

    def test_rejects_non_numeric_polygon_point(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "polygon": [["a", 0], [0.5, 0.5], [0.5, 1]]}], CLASS_NAMES)

    def test_rejects_empty_or_missing_classes(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "classes": []}], CLASS_NAMES)
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "classes": None}], CLASS_NAMES)

    def test_rejects_unknown_class_name(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([zone(classes=('unicorn',))], CLASS_NAMES)

    def test_class_names_are_case_insensitive_and_deduped(self):
        parsed = count_zones.parse_zones([zone(classes=('PERSON', 'person', 'Bicycle'))], CLASS_NAMES)
        self.assertEqual(parsed[0]["classes"], ["person", "bicycle"])

    def test_rejects_bad_metric(self):
        with self.assertRaises(ValueError):
            count_zones.parse_zones([{**zone(), "metric": "bogus"}], CLASS_NAMES)

    def test_skips_class_validation_when_no_vocabulary_given(self):
        parsed = count_zones.parse_zones([zone(classes=('unicorn',))])
        self.assertEqual(parsed[0]["classes"], ["unicorn"])


class FootpointAndPolygonTests(unittest.TestCase):
    def test_footpoint_is_bottom_centre(self):
        x, y = count_zones.footpoint([0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(x, 0.3)
        self.assertAlmostEqual(y, 0.5)

    def test_point_in_polygon_inside_and_outside(self):
        self.assertTrue(count_zones._point_in_polygon((0.5, 0.5), SQUARE))
        self.assertFalse(count_zones._point_in_polygon((0.1, 0.1), SQUARE))


def box_at(cx, cy, half=0.02):
    """A tiny box whose footpoint sits at (cx, cy)."""
    return (cx - half, cy - half, cx + half, cy)


class ZoneCounterEntryExitTests(unittest.TestCase):
    def setUp(self):
        self.counter = make_counter([zone()], now=1_000_000.0)

    def snap(self, now):
        by_id = {z['id']: z for z in self.counter.snapshot(now)}
        return by_id['z1']

    def test_entry_requires_continuous_half_second_inside(self):
        t0 = 1_000_000.0
        self.counter.update(t0, [(1, 'person', box_at(0.5, 0.5))])
        self.assertEqual(self.snap(t0)['stats']['person']['entered'], 0)
        self.counter.update(t0 + 0.3, [(1, 'person', box_at(0.5, 0.5))])
        self.assertEqual(self.snap(t0 + 0.3)['stats']['person']['entered'], 0)
        self.counter.update(t0 + 0.6, [(1, 'person', box_at(0.5, 0.5))])
        snap = self.snap(t0 + 0.6)
        self.assertEqual(snap['stats']['person']['entered'], 1)
        self.assertEqual(len(snap['inside_now']), 1)
        self.assertEqual(snap['inside_now'][0]['track_id'], 1)

    def test_flicker_at_the_edge_does_not_double_count(self):
        t0 = 1_000_000.0
        # Inside, briefly outside (resets the entry timer), inside again,
        # this time long enough -> exactly one entry.
        self.counter.update(t0, [(1, 'person', box_at(0.5, 0.5))])
        self.counter.update(t0 + 0.2, [(1, 'person', box_at(0.05, 0.05))])  # outside
        self.counter.update(t0 + 0.3, [(1, 'person', box_at(0.5, 0.5))])
        self.counter.update(t0 + 0.9, [(1, 'person', box_at(0.5, 0.5))])
        self.assertEqual(self.snap(t0 + 0.9)['stats']['person']['entered'], 1)

    def test_exit_after_half_second_continuously_outside(self):
        t0 = 1_000_000.0
        self.counter.update(t0, [(1, 'person', box_at(0.5, 0.5))])
        self.counter.update(t0 + 0.6, [(1, 'person', box_at(0.5, 0.5))])  # confirmed inside
        self.assertEqual(self.snap(t0 + 0.6)['stats']['person']['entered'], 1)
        self.counter.update(t0 + 0.7, [(1, 'person', box_at(0.05, 0.05))])  # now outside
        self.assertEqual(self.snap(t0 + 0.7)['stats']['person']['exited'], 0)
        self.counter.update(t0 + 1.3, [(1, 'person', box_at(0.05, 0.05))])  # 0.6s outside
        snap = self.snap(t0 + 1.3)
        self.assertEqual(snap['stats']['person']['exited'], 1)
        self.assertEqual(snap['inside_now'], [])

    def test_exit_by_lost_timeout_passes_zone(self):
        t0 = 1_000_000.0
        self.counter.update(t0, [(1, 'person', box_at(0.5, 0.5))])
        self.counter.update(t0 + 0.6, [(1, 'person', box_at(0.5, 0.5))])
        self.assertEqual(self.snap(t0 + 0.6)['stats']['person']['entered'], 1)
        # Track vanishes entirely (no detection at all) for >= 3s.
        snap = self.snap(t0 + 0.6 + count_zones.DEFAULT_LOST_TIMEOUT + 0.1)
        self.counter.update(t0 + 0.6 + count_zones.DEFAULT_LOST_TIMEOUT + 0.1, [])
        snap = self.snap(t0 + 0.6 + count_zones.DEFAULT_LOST_TIMEOUT + 0.1)
        self.assertEqual(snap['stats']['person']['exited'], 1)

    def test_dwell_zone_uses_longer_lost_timeout(self):
        z = zone(metric='dwell', classes=('car',))
        counter = make_counter([z], now=1_000_000.0)
        t0 = 1_000_000.0
        counter.update(t0, [(1, 'car', box_at(0.5, 0.5))])
        counter.update(t0 + 0.6, [(1, 'car', box_at(0.5, 0.5))])
        # Gone for longer than the passes timeout but less than the dwell one:
        # still counted as inside.
        gap = t0 + 0.6 + count_zones.DEFAULT_LOST_TIMEOUT + 1
        counter.update(gap, [])
        snap = {z['id']: z for z in counter.snapshot(gap)}['z1']
        self.assertEqual(snap['stats']['car']['exited'], 0)
        self.assertEqual(len(snap['inside_now']), 1)
        # Past the dwell timeout it is held as a lost stay (it may be re-found
        # at the same spot), still shown inside...
        gone = t0 + 0.6 + count_zones.DWELL_LOST_TIMEOUT + 1
        counter.update(gone, [])
        snap = {z['id']: z for z in counter.snapshot(gone)}['z1']
        self.assertEqual(snap['stats']['car']['exited'], 0)
        self.assertEqual(len(snap['inside_now']), 1)
        # ...and recorded as exited once nobody resumes it within the window.
        later = gone + count_zones.RESUME_WINDOW + 1
        counter.update(later, [])
        snap = {z['id']: z for z in counter.snapshot(later)}['z1']
        self.assertEqual(snap['stats']['car']['exited'], 1)

    def test_only_matching_class_is_counted(self):
        t0 = 1_000_000.0
        self.counter.update(t0, [(1, 'dog', box_at(0.5, 0.5))])
        self.counter.update(t0 + 0.6, [(1, 'dog', box_at(0.5, 0.5))])
        snap = self.snap(t0 + 0.6)
        self.assertNotIn('dog', snap['stats'])  # zone only asked for 'person'

    def test_child_counts_as_person_when_zone_asks_for_person(self):
        t0 = 1_000_000.0
        self.counter.update(t0, [(1, 'child', box_at(0.5, 0.5))])
        self.counter.update(t0 + 0.6, [(1, 'child', box_at(0.5, 0.5))])
        snap = self.snap(t0 + 0.6)
        self.assertEqual(snap['stats']['person']['entered'], 1)


class DwellStatsTests(unittest.TestCase):
    def test_dwell_average_and_max(self):
        z = zone(metric='dwell', classes=('car',))
        counter = make_counter([z], now=0.0)
        # Track 1: entry sample at t=0 (first inside), last seen inside at
        # t=9.9, then outside from t=10.0, exit confirmed at t=10.6 -> dwell
        # (last-inside-seen minus entry time) == 9.9s.
        counter.update(0.0, [(1, 'car', box_at(0.5, 0.5))])
        counter.update(0.5, [(1, 'car', box_at(0.5, 0.5))])
        counter.update(9.9, [(1, 'car', box_at(0.5, 0.5))])
        counter.update(10.0, [(1, 'car', box_at(0.05, 0.05))])
        counter.update(10.6 + count_zones.DWELL_EXIT_HYSTERESIS, [(1, 'car', box_at(0.05, 0.05))])  # exit confirmed
        # Track 2: entry sample at t=20, last seen inside at t=25.4, exits at
        # t=26.1 -> dwell 5.4s.
        counter.update(40.0, [(2, 'car', box_at(0.5, 0.5))])
        counter.update(40.5, [(2, 'car', box_at(0.5, 0.5))])
        counter.update(45.4, [(2, 'car', box_at(0.5, 0.5))])
        counter.update(45.5, [(2, 'car', box_at(0.05, 0.05))])
        counter.update(46.1 + count_zones.DWELL_EXIT_HYSTERESIS, [(2, 'car', box_at(0.05, 0.05))])
        snap = {zz['id']: zz for zz in counter.snapshot(80.0)}['z1']
        stats = snap['stats']['car']
        self.assertEqual(stats['dwell_count'], 2)
        self.assertAlmostEqual(stats['dwell_max'], 9.9, places=1)
        self.assertAlmostEqual(stats['dwell_avg'], (9.9 + 5.4) / 2, places=1)


class PassesCountingTests(unittest.TestCase):
    def test_passes_counts_per_class(self):
        z = zone(classes=('bicycle', 'motorcycle'), metric='passes')
        counter = make_counter([z], now=0.0)
        counter.update(0.0, [(1, 'bicycle', box_at(0.5, 0.5)), (2, 'motorcycle', box_at(0.3, 0.3))])
        counter.update(0.6, [(1, 'bicycle', box_at(0.5, 0.5)), (2, 'motorcycle', box_at(0.3, 0.3))])
        snap = {zz['id']: zz for zz in counter.snapshot(0.6)}['z1']
        self.assertEqual(snap['stats']['bicycle']['entered'], 1)
        self.assertEqual(snap['stats']['motorcycle']['entered'], 1)


class MidnightRolloverTests(unittest.TestCase):
    def test_rollover_resets_day_stats(self):
        # 2024-01-01 23:59:00 local time.
        import datetime
        day1 = datetime.datetime(2024, 1, 1, 23, 59, 0).timestamp()
        counter = make_counter([zone()], now=day1 - 60)
        day2_early = datetime.datetime(2024, 1, 2, 0, 0, 30).timestamp()
        counter.update(day1, [(1, 'person', box_at(0.5, 0.5))])
        counter.update(day1 + 0.6, [(1, 'person', box_at(0.5, 0.5))])
        self.assertEqual(counter.to_dict()['zones']['z1']['person']['entered'], 1)
        # Crossing midnight resets stats for the new day.
        counter.update(day2_early, [(2, 'person', box_at(0.5, 0.5))])
        self.assertEqual(counter.to_dict()['date'], '2024-01-02')
        self.assertEqual(counter.to_dict()['zones'].get('z1', {}).get('person', {}).get('entered', 0), 0)


class SaveLoadTests(unittest.TestCase):
    def test_round_trip(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'zones.json'
            counter = make_counter([zone()], now=1000.0)
            counter.update(1000.0, [(1, 'person', box_at(0.5, 0.5))])
            counter.update(1000.6, [(1, 'person', box_at(0.5, 0.5))])
            counter.update(1001.3, [(1, 'person', box_at(0.05, 0.05))])  # exits
            counter.save(path)

            reloaded = make_counter([zone()], now=1000.0)
            reloaded.load(path)
            self.assertEqual(reloaded.to_dict(), counter.to_dict())

    def test_stale_day_file_is_ignored(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'zones.json'
            path.write_text(json.dumps({"date": "2000-01-01", "zones": {"z1": {"person": {"entered": 99, "exited": 0, "dwell_count": 0, "dwell_sum": 0.0, "dwell_max": 0.0}}}}))
            counter = make_counter([zone()], now=1_700_000_000.0)
            counter.load(path)
            self.assertEqual(counter.to_dict()['zones'], {})


class ReconfigureKeepsStatsTests(unittest.TestCase):
    def test_unchanged_zone_id_keeps_stats_changed_id_does_not(self):
        z1 = zone(id='z1', name='Yard')
        z2 = zone(id='z2', name='Porch', classes=('dog',))
        counter = make_counter([z1, z2], now=0.0)
        counter.update(0.0, [(1, 'person', box_at(0.5, 0.5))])
        counter.update(0.6, [(1, 'person', box_at(0.5, 0.5))])
        self.assertEqual(counter.to_dict()['zones']['z1']['person']['entered'], 1)

        # Reconfigure: z1 kept as-is (same id), z2 replaced by a new id z3.
        z3 = zone(id='z3', name='New porch', classes=('dog',))
        counter.reconfigure([z1, z3])
        snapshot_ids = {z['id'] for z in counter.snapshot(1.0)}
        self.assertEqual(snapshot_ids, {'z1', 'z3'})
        self.assertEqual(counter.to_dict()['zones']['z1']['person']['entered'], 1)


def _extract_edit_settings_handler():
    """Same technique as test_ignore_areas.py: pull just the /edit_settings
    branch out of clearcam.py's do_GET and run it standalone."""
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
    namespace = {'json': json, 'ignore_areas_mod': __import__('utils.ignore_areas', fromlist=['x']),
                 'count_zones_mod': count_zones, 'class_labels': CLASS_NAMES}
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


class EditSettingsCountZonesHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = _extract_edit_settings_handler()

    def _call(self, cam_settings, count_zones_param):
        self_obj = FakeSelf(cam_settings)
        database_ns = self_obj._db()
        parsed_path = SimpleNamespace(path='/edit_settings')
        query = {'count_zones': [count_zones_param]} if count_zones_param is not None else {}
        func = self.handler
        func.__globals__['database'] = database_ns
        func(self_obj, parsed_path, query, 'tapoc113')
        return self_obj

    def test_sets_count_zones(self):
        payload = json.dumps([zone()])
        result = self._call({}, payload)
        self.assertIsNone(result.refusal)
        self.assertEqual(len(result._settings['count_zones']), 1)
        self.assertEqual(result._settings['count_zones'][0]['id'], 'z1')
        self.assertEqual(result.responses, [200])

    def test_invalid_payload_is_refused_and_not_persisted(self):
        payload = json.dumps([{**zone(), "metric": "bogus"}])
        result = self._call({'threshold': 0.4}, payload)
        self.assertIsNotNone(result.refusal)
        self.assertEqual(result.refusal[0], 400)
        self.assertEqual(result._settings, {'threshold': 0.4})

    def test_empty_list_clears_previously_stored_zones(self):
        result = self._call({'count_zones': [zone()]}, json.dumps([]))
        self.assertIsNone(result.refusal)
        self.assertNotIn('count_zones', result._settings)

    def test_no_count_zones_param_leaves_settings_untouched(self):
        result = self._call({'threshold': 0.5}, None)
        self.assertIsNone(result.refusal)
        self.assertNotIn('count_zones', result._settings)
        self.assertEqual(result._settings['threshold'], 0.5)


if __name__ == '__main__':
    unittest.main()


class StartupAndResumeTests(unittest.TestCase):
    def dwell_zone(self):
        return zone(classes=('car',), metric='dwell')

    def test_object_present_at_startup_is_timed_not_counted(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, min_park=0.0)
        for t in (1.0, 2.0, 30.0):
            c.update(t, [(1, 'car', box_at(0.5, 0.5))])
        snap = c.snapshot(30.0)[0]
        self.assertEqual(snap['stats']['car']['entered'], 0)
        self.assertEqual(len(snap['inside_now']), 1)
        self.assertTrue(snap['inside_now'][0]['adopted'])

    def test_arrival_after_grace_counts(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, min_park=0.0)
        c.update(1.0, [])                                   # counting starts, zone empty
        c.update(100.0, [(1, 'car', box_at(0.5, 0.5))])
        c.update(101.0, [(1, 'car', box_at(0.5, 0.5))])
        self.assertEqual(c.snapshot(101.0)[0]['stats']['car']['entered'], 1)

    def test_lost_and_refound_at_same_spot_is_one_stay(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, startup_grace=0.0, min_park=0.0)
        c.update(10.0, [(1, 'car', box_at(0.5, 0.5))]); c.update(11.0, [(1, 'car', box_at(0.5, 0.5))])
        c.update(100.0, [])                         # track lost (> dwell lost timeout)
        self.assertEqual(len(c.snapshot(100.0)[0]['inside_now']), 1)   # still shown as parked
        c.update(300.0, [(2, 'car', box_at(0.505, 0.5))]); c.update(301.0, [(2, 'car', box_at(0.505, 0.5))])
        snap = c.snapshot(301.0)[0]
        self.assertEqual(snap['stats']['car']['entered'], 1)
        self.assertEqual(snap['inside_now'][0]['since'], 10.0)

    def test_lost_stay_not_resumed_is_recorded_when_window_expires(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, startup_grace=0.0, min_park=0.0)
        c.update(10.0, [(1, 'car', box_at(0.5, 0.5))]); c.update(70.0, [(1, 'car', box_at(0.5, 0.5))])
        c.update(100.0, [])
        c.update(100.0 + count_zones.RESUME_WINDOW + 80, [])
        s = c.snapshot(3000.0)[0]
        self.assertEqual(s['stats']['car']['exited'], 1)
        self.assertAlmostEqual(s['stats']['car']['dwell_max'], 60.0)
        self.assertEqual(s['inside_now'], [])

    def test_different_spot_is_a_new_car(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, startup_grace=0.0, min_park=0.0)
        c.update(10.0, [(1, 'car', box_at(0.4, 0.5))]); c.update(11.0, [(1, 'car', box_at(0.4, 0.5))])
        c.update(100.0, [])
        c.update(300.0, [(2, 'car', box_at(0.6, 0.5))]); c.update(301.0, [(2, 'car', box_at(0.6, 0.5))])
        self.assertEqual(c.snapshot(301.0)[0]['stats']['car']['entered'], 2)


class DwellEdgeWobbleTests(unittest.TestCase):
    def test_parked_car_wobbling_over_the_edge_is_one_stay(self):
        z = zone(classes=('car',), metric='dwell')
        c = make_counter([z], now=0.0)
        inside, outside = box_at(0.5, 0.5), box_at(0.5, 0.5 + 0.5)   # footpoint jumps out of the square
        t = 0.0
        for k in range(40):   # 20 s alternating in/out every 2 s
            c.update(t, [(1, 'car', inside if (k // 2) % 2 == 0 else outside)])
            t += 0.5
        snap = c.snapshot(t)[0]
        self.assertEqual(snap['stats']['car']['entered'], 1)
        self.assertEqual(snap['stats']['car']['exited'], 0)


class RiderTests(unittest.TestCase):
    def test_cyclist_is_not_a_person_walking(self):
        c = make_counter([zone(classes=('person',)), zone(id='z2', classes=('bicycle',))], now=0.0)
        person, bike = (0.48, 0.40, 0.52, 0.55), (0.46, 0.48, 0.54, 0.58)
        for t in (0.0, 0.6, 1.2):
            c.update(t, [(1, 'person', person), (2, 'bicycle', bike)])
        c.update(1.8, [(1, 'person', person)])          # bike missed this frame: still a rider
        snap = {z['id']: z for z in c.snapshot(2.0)}
        self.assertEqual(snap['z1']['stats']['person']['entered'], 0)
        self.assertEqual(snap['z2']['stats']['bicycle']['entered'], 1)

    def test_pedestrian_next_to_parked_bike_still_counts(self):
        c = make_counter([zone(classes=('person',))], now=0.0)
        person, bike = (0.30, 0.40, 0.34, 0.55), (0.60, 0.48, 0.68, 0.58)
        for t in (0.0, 0.6):
            c.update(t, [(1, 'person', person), (2, 'bicycle', bike)])
        self.assertEqual(c.snapshot(1.0)[0]['stats']['person']['entered'], 1)


class PersonDwellTests(unittest.TestCase):
    def test_person_lost_from_view_is_not_held_as_inside(self):
        c = make_counter([zone(classes=('person',), metric='dwell')], now=0.0)
        for t in (0.0, 0.6, 3.0):
            c.update(t, [(1, 'person', box_at(0.5, 0.5))])
        c.update(3.0 + count_zones.PERSON_DWELL_LOST_TIMEOUT + 0.1, [])
        snap = c.snapshot(20.0)[0]
        self.assertEqual(snap['inside_now'], [])
        self.assertEqual(snap['stats']['person']['dwell_count'], 1)
        self.assertAlmostEqual(snap['stats']['person']['dwell_max'], 3.0)


class DwellTimerTests(unittest.TestCase):
    def test_timers_for_tracks_inside_dwell_zones_only(self):
        c = make_counter([zone(classes=('car',), metric='dwell'), zone(id='p', classes=('person',))], now=0.0)
        for t in (10.0, 10.6):
            c.update(t, [(1, 'car', box_at(0.5, 0.5)), (2, 'person', box_at(0.5, 0.5))])
        timers = c.dwell_timers(70.0)
        self.assertEqual(set(timers), {1})                 # the person's zone counts passes, no timer
        self.assertAlmostEqual(timers[1][0], 60.0)
        self.assertFalse(timers[1][1])


class LateStartTests(unittest.TestCase):
    def test_grace_starts_at_first_frame_not_at_creation(self):
        # Models and streams take a while: the first frame arrives long after
        # the counter is created. Cars already there must still be adopted.
        c = count_zones.ZoneCounter([zone(classes=('car',), metric='dwell')], now=0.0)
        for t in (120.0, 121.0, 150.0):
            c.update(t, [(1, 'car', box_at(0.5, 0.5))])
        snap = c.snapshot(150.0)[0]
        self.assertEqual(snap['stats']['car']['entered'], 0)
        self.assertTrue(snap['inside_now'][0]['adopted'])


class RiderSpeedTests(unittest.TestCase):
    def run_track(self, step_per_second, frames=12):
        c = make_counter([zone(classes=('person',))], now=0.0)
        for k in range(frames):
            t = k * 0.1
            x = 0.21 + step_per_second * t
            c.update(t, [(1, 'person', (x, 0.40, x + 0.02, 0.50))])   # box height 0.10
        return c.snapshot(2.0)[0]['stats']['person']['entered']

    def test_fast_person_without_a_detected_bike_is_a_rider(self):
        self.assertEqual(self.run_track(0.40), 0)     # 4 heights/s: cycling

    def test_walker_counts(self):
        self.assertEqual(self.run_track(0.08), 1)     # 0.8 heights/s: walking


class ParkingStayTests(unittest.TestCase):
    """tapo360's parking zones: a car must stay a minute to count as parked,
    and a restart must not lose a parked car's real arrival time."""
    def dwell_zone(self):
        return zone(id='bays', classes=('car',), metric='dwell')

    def park(self, c, track, t0, t1, at=(0.5, 0.5), step=5.0):
        t = t0
        while t <= t1:
            c.update(t, [(track, 'car', box_at(*at))]); t += step

    def leave(self, c, track, t):
        c.update(t, [(track, 'car', box_at(0.05, 0.05))])
        c.update(t + count_zones.DWELL_EXIT_HYSTERESIS + 1, [(track, 'car', box_at(0.05, 0.05))])

    def test_car_driving_through_is_not_a_parking_stay(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, startup_grace=0.0)
        self.park(c, 1, 10.0, 14.0, step=1.0)          # 4 s across the bays
        self.leave(c, 1, 15.0)
        stats = c.snapshot(100.0)[0]['stats']['car']
        self.assertEqual((stats['entered'], stats['exited'], stats['dwell_count']), (0, 0, 0))

    def test_car_that_stays_counts_once_with_its_full_stay(self):
        c = count_zones.ZoneCounter([self.dwell_zone()], now=0.0, startup_grace=0.0)
        self.park(c, 1, 10.0, 30.0)
        self.assertEqual(c.snapshot(30.0)[0]['stats']['car']['entered'], 0)    # not yet a minute
        self.park(c, 1, 35.0, 910.0)
        self.assertEqual(c.snapshot(910.0)[0]['stats']['car']['entered'], 1)
        self.leave(c, 1, 915.0)
        stats = c.snapshot(1000.0)[0]['stats']['car']
        self.assertEqual((stats['entered'], stats['exited'], stats['dwell_count']), (1, 1, 1))
        self.assertAlmostEqual(stats['dwell_max'], 900.0, delta=5)

    def test_restart_keeps_a_parked_cars_arrival_time(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'zones.json'
            day = 1_700_000_000.0
            c = count_zones.ZoneCounter([self.dwell_zone()], now=day, startup_grace=0.0)
            self.park(c, 1, day + 10, day + 600)        # parked at day+10
            c.save(path)
            # ClearCam restarts: new counter, new tracker ids, zones arrive after load
            r = count_zones.ZoneCounter(now=day + 700)
            r.load(path)
            r.reconfigure([self.dwell_zone()])
            self.park(r, 7, day + 720, day + 3600)      # same spot, new track id, inside the grace period
            snap = r.snapshot(day + 3600)[0]
            self.assertEqual(len(snap['inside_now']), 1)
            self.assertFalse(snap['inside_now'][0]['adopted'])
            self.assertAlmostEqual(snap['inside_now'][0]['since'], day + 10, delta=1)
            self.leave(r, 7, day + 3605)
            stats = r.snapshot(day + 3700)[0]['stats']['car']
            self.assertEqual(stats['entered'], 1)                                # not counted twice
            self.assertAlmostEqual(stats['dwell_max'], 3590.0, delta=10)        # whole stay, across the restart

    def test_restart_with_the_car_gone_does_not_invent_a_stay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'zones.json'
            day = 1_700_000_000.0
            c = count_zones.ZoneCounter([self.dwell_zone()], now=day, startup_grace=0.0)
            self.park(c, 1, day + 10, day + 600)
            c.save(path)
            r = count_zones.ZoneCounter(now=day + 700)
            r.load(path)
            r.reconfigure([self.dwell_zone()])
            r.update(day + 720, [])
            r.update(day + 720 + count_zones.RESUME_WINDOW + 10, [])         # never seen again
            stats = r.snapshot(day + 720 + count_zones.RESUME_WINDOW + 10)[0]['stats']['car']
            self.assertEqual((stats['entered'], stats['exited']), (1, 1))      # the stay before the restart
            self.assertAlmostEqual(stats['dwell_max'], 590.0, delta=10)

    def test_parked_cars_survive_a_restart_after_midnight(self):
        import datetime
        evening = datetime.datetime(2024, 3, 1, 23, 40, 0).timestamp()
        after_midnight = datetime.datetime(2024, 3, 2, 0, 5, 0).timestamp()
        c = count_zones.ZoneCounter([self.dwell_zone()], now=evening, startup_grace=0.0)
        self.park(c, 1, evening, evening + 19 * 60)          # seen until 23:59
        saved = c.to_dict()                                  # last save went to yesterday's file
        r = count_zones.ZoneCounter(now=after_midnight)
        r.load_dict(saved)
        r.reconfigure([self.dwell_zone()])
        self.park(r, 9, after_midnight + 5, after_midnight + 300)
        inside = r.snapshot(after_midnight + 300)[0]['inside_now']
        self.assertEqual(len(inside), 1)
        self.assertAlmostEqual(inside[0]['since'], evening, delta=1)
        self.assertEqual(r.to_dict()['zones'].get('bays', {}).get('car', {}).get('entered', 0), 0)  # yesterday's stats not merged

    def test_a_car_gone_after_restart_is_not_shown_as_parked(self):
        day = 1_700_000_000.0
        c = count_zones.ZoneCounter([self.dwell_zone()], now=day, startup_grace=0.0)
        self.park(c, 1, day + 10, day + 600)
        r = count_zones.ZoneCounter(now=day + 700)
        r.load_dict(c.to_dict()); r.reconfigure([self.dwell_zone()])
        r.update(day + 760, [])
        self.assertEqual(r.snapshot(day + 760)[0]['inside_now'], [])

