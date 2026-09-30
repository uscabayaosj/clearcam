import io
import json
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import roboflow_sync as rs
from script import roboflow_dataset as rd

TARGET = rd.TARGET_CLASSES  # ['person', 'bicycle', 'car', 'truck', 'dog', 'stroller', 'child', 'scooter']


# --------------------------------------------------------------------- specs

class ParseSourceSpecTests(unittest.TestCase):
    def test_workspace_project_only(self):
        self.assertEqual(rd.parse_source_spec('marcu/stroller-vxfbx'), ('marcu', 'stroller-vxfbx', None))

    def test_workspace_project_version(self):
        self.assertEqual(rd.parse_source_spec('ws/proj:3'), ('ws', 'proj', 3))

    def test_missing_slash_raises(self):
        with self.assertRaises(ValueError):
            rd.parse_source_spec('just-a-project:3')

    def test_non_integer_version_raises(self):
        with self.assertRaises(ValueError):
            rd.parse_source_spec('ws/proj:v3')

    def test_empty_version_after_colon_is_none(self):
        self.assertEqual(rd.parse_source_spec('ws/proj:'), ('ws', 'proj', None))


class SourceSlugTests(unittest.TestCase):
    def test_slug_from_workspace_and_project(self):
        self.assertEqual(rd.source_slug('marcu', 'stroller-vxfbx'), 'marcu-stroller-vxfbx')

    def test_slug_sanitises_odd_characters(self):
        self.assertEqual(rd.source_slug('Whitera1313its Workspace', 'scooter-yhjgq'),
                          'whitera1313its-workspace-scooter-yhjgq')


class PickLatestVersionTests(unittest.TestCase):
    def test_picks_highest_from_ids(self):
        payload = dict(versions=[dict(id='ws/proj/1'), dict(id='ws/proj/3'), dict(id='ws/proj/2')])
        self.assertEqual(rd.pick_latest_version(payload), 3)

    def test_falls_back_to_version_field(self):
        payload = dict(versions=[dict(version=2), dict(version=5)])
        self.assertEqual(rd.pick_latest_version(payload), 5)

    def test_tolerates_malformed_items(self):
        payload = dict(versions=[dict(id='ws/proj/x'), 'not-a-dict', dict(id='ws/proj/4')])
        self.assertEqual(rd.pick_latest_version(payload), 4)

    def test_missing_versions_raises(self):
        with self.assertRaises(rs.RoboflowError):
            rd.pick_latest_version(dict())

    def test_empty_versions_raises(self):
        with self.assertRaises(rs.RoboflowError):
            rd.pick_latest_version(dict(versions=[]))


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._body


class FetchLatestVersionTests(unittest.TestCase):
    def test_success(self):
        calls = []

        def opener(request, timeout=30):
            calls.append(request.full_url)
            return FakeResponse(dict(versions=[dict(id='ws/proj/1'), dict(id='ws/proj/7')]))

        version = rd.fetch_latest_version('ws', 'proj', 'secretkey', opener=opener)
        self.assertEqual(version, 7)
        self.assertIn('/ws/proj?', calls[0])
        self.assertIn('api_key=secretkey', calls[0])

    def test_http_error_redacts_key(self):
        err = urllib.error.HTTPError('url', 403, 'forbidden', {}, io.BytesIO(b'bad key secretkey here'))

        def opener(request, timeout=30):
            raise err

        with self.assertRaises(rs.RoboflowError) as ctx:
            rd.fetch_latest_version('ws', 'proj', 'secretkey', opener=opener)
        self.assertNotIn('secretkey', str(ctx.exception))


# ---------------------------------------------------------------- subsampling

class DeterministicSubsampleTests(unittest.TestCase):
    def test_deterministic_regardless_of_input_order(self):
        names = [f'img{i}.jpg' for i in range(50)]
        a = rd.deterministic_subsample(names, 10)
        b = rd.deterministic_subsample(list(reversed(names)), 10)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)

    def test_zero_count_is_empty(self):
        self.assertEqual(rd.deterministic_subsample(['a.jpg', 'b.jpg'], 0), [])

    def test_count_above_length_returns_all_ranked(self):
        names = ['a.jpg', 'b.jpg']
        out = rd.deterministic_subsample(names, 10)
        self.assertEqual(set(out), set(names))
        self.assertEqual(len(out), 2)


class SelectSourceImagesTests(unittest.TestCase):
    def test_val_capped_at_fraction_of_max_images(self):
        train_names = [f't{i}.jpg' for i in range(1000)]
        valid_names = [f'v{i}.jpg' for i in range(1000)]
        train_sel, val_sel = rd.select_source_images(train_names, valid_names, max_images=100, val_fraction=0.15)
        self.assertEqual(len(val_sel), 15)  # capped at 15% of 100
        self.assertEqual(len(train_sel), 85)  # remaining budget

    def test_small_source_never_exceeds_available_images(self):
        train_names = ['t1.jpg', 't2.jpg']
        valid_names = ['v1.jpg']
        train_sel, val_sel = rd.select_source_images(train_names, valid_names, max_images=1500)
        self.assertEqual(set(train_sel), {'t1.jpg', 't2.jpg'})
        self.assertEqual(set(val_sel), {'v1.jpg'})

    def test_reruns_are_deterministic(self):
        train_names = [f't{i}.jpg' for i in range(300)]
        valid_names = [f'v{i}.jpg' for i in range(60)]
        a = rd.select_source_images(train_names, valid_names, max_images=100)
        b = rd.select_source_images(list(reversed(train_names)), list(reversed(valid_names)), max_images=100)
        self.assertEqual(a, b)


# -------------------------------------------------------------- class mapping

class ClassMappingTests(unittest.TestCase):
    def test_maps_into_the_eight_targets(self):
        src = ['0-human', 'Stroller', 'kids', 'kickboard', 'car', 'truck', 'dog', 'bicycle']
        index_map = rs.build_index_map(src, TARGET)
        self.assertEqual(TARGET[index_map[0]], 'person')       # 0-human
        self.assertEqual(TARGET[index_map[1]], 'stroller')      # Stroller
        self.assertEqual(TARGET[index_map[2]], 'child')         # kids
        self.assertEqual(TARGET[index_map[3]], 'scooter')       # kickboard
        self.assertEqual(TARGET[index_map[4]], 'car')
        self.assertEqual(TARGET[index_map[5]], 'truck')
        self.assertEqual(TARGET[index_map[6]], 'dog')
        self.assertEqual(TARGET[index_map[7]], 'bicycle')

    def test_wheelchair_is_dropped(self):
        src = ['Wheelchair', 'person']
        index_map = rs.build_index_map(src, TARGET)
        self.assertNotIn(0, index_map)
        self.assertEqual(TARGET[index_map[1]], 'person')

    def test_additional_scooter_and_stroller_aliases(self):
        for raw in ('kickboard', 'electric-kickboard', 'kick-scooter', 'electric scooter'):
            self.assertEqual(rs.normalise_class(raw, TARGET), 'scooter', raw)
        for raw in ('prams', 'carriage'):
            self.assertEqual(rs.normalise_class(raw, TARGET), 'stroller', raw)

    def test_remap_lines_drops_unmapped_and_reindexes(self):
        index_map = {0: TARGET.index('person'), 2: TARGET.index('dog')}
        text = '0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n2 0.4 0.4 0.1 0.1\n'
        out = rd.remap_lines(text, index_map)
        self.assertEqual(out, [f'{TARGET.index("person")} 0.5 0.5 0.2 0.2',
                                f'{TARGET.index("dog")} 0.4 0.4 0.1 0.1'])


# ------------------------------------------------------------------- VOC XML

class VocXmlFromYoloLinesTests(unittest.TestCase):
    def test_builds_boxes_with_class_names(self):
        lines = [f'{TARGET.index("person")} 0.5 0.5 0.2 0.2']
        xml = rs.voc_xml_from_yolo_lines(lines, TARGET, 'img.jpg', 100, 100)
        self.assertIn('<filename>img.jpg</filename>', xml)
        self.assertIn('<name>person</name>', xml)
        self.assertIn('<width>100</width>', xml)
        self.assertIn('<height>100</height>', xml)
        self.assertIn('<xmin>40</xmin>', xml)
        self.assertIn('<ymin>40</ymin>', xml)
        self.assertIn('<xmax>60</xmax>', xml)
        self.assertIn('<ymax>60</ymax>', xml)

    def test_empty_lines_produce_no_objects(self):
        xml = rs.voc_xml_from_yolo_lines([], TARGET, 'img.jpg', 50, 50)
        self.assertNotIn('<object>', xml)

    def test_out_of_range_class_is_skipped(self):
        xml = rs.voc_xml_from_yolo_lines(['99 0.5 0.5 0.1 0.1'], TARGET, 'img.jpg', 50, 50)
        self.assertNotIn('<object>', xml)

    def test_escapes_filename(self):
        xml = rs.voc_xml_from_yolo_lines([], TARGET, 'a & b.jpg', 10, 10)
        self.assertIn('a &amp; b.jpg', xml)


# --------------------------------------------------------------------- ledger

class LedgerTests(unittest.TestCase):
    def test_empty_ledger_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rd.load_ledger(tmp), set())

    def test_append_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd.append_ledger(tmp, 'src_a.jpg', 'id-1')
            rd.append_ledger(tmp, 'src_b.jpg', 'id-2')
            self.assertEqual(rd.load_ledger(tmp), {'src_a.jpg', 'src_b.jpg'})

    def test_resume_skips_already_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd.append_ledger(tmp, 'src_a.jpg', 'id-1')
            already = rd.load_ledger(tmp)
            all_records = [dict(final_name='src_a.jpg'), dict(final_name='src_b.jpg')]
            todo = [r for r in all_records if r['final_name'] not in already]
            self.assertEqual([r['final_name'] for r in todo], ['src_b.jpg'])

    def test_ignores_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = rd.ledger_path(tmp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('not json\n' + json.dumps(dict(filename='ok.jpg', id='1')) + '\n')
            self.assertEqual(rd.load_ledger(tmp), {'ok.jpg'})


# --------------------------------------------------------------------- upload

class FakeOpener:
    """Sequential fake responses/exceptions, one per opener() call."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, request, timeout=30):
        self.calls.append(request.full_url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def http_error(code, body=b'server exploded'):
    return urllib.error.HTTPError('url', code, 'err', {}, io.BytesIO(body))


class UploadWithRetryTests(unittest.TestCase):
    def setUp(self):
        self._orig_sleep = time.sleep
        time.sleep = lambda *_a, **_k: None  # keep the retry tests fast

    def tearDown(self):
        time.sleep = self._orig_sleep

    def test_retries_transient_500_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'a.jpg'
            image.write_bytes(b'jpg-bytes')
            cfg = dict(api_key='secretkey', project='proj')
            opener = FakeOpener([
                http_error(500),
                dict(success=True, id='id-a'), dict(success=True),
            ])
            result = rd.upload_with_retry(cfg, image, '<annotation/>', 'train', 'universe:ws/proj', opener=opener)
            self.assertEqual(result['id'], 'id-a')
            self.assertEqual(len(opener.calls), 3)

    def test_non_transient_error_raises_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'a.jpg'
            image.write_bytes(b'jpg-bytes')
            cfg = dict(api_key='secretkey', project='proj')
            opener = FakeOpener([http_error(400, b'bad request secretkey')])
            with self.assertRaises(rs.RoboflowError) as ctx:
                rd.upload_with_retry(cfg, image, '<annotation/>', 'train', 'universe:ws/proj', opener=opener)
            self.assertEqual(len(opener.calls), 1)
            self.assertNotIn('secretkey', str(ctx.exception))

    def test_gives_up_after_max_tries(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'a.jpg'
            image.write_bytes(b'jpg-bytes')
            cfg = dict(api_key='secretkey', project='proj')
            opener = FakeOpener([http_error(503) for _ in range(5)])
            with self.assertRaises(rs.RoboflowError):
                rd.upload_with_retry(cfg, image, '<annotation/>', 'train', 'universe:ws/proj',
                                      opener=opener, max_tries=5)
            self.assertEqual(len(opener.calls), 5)

    def test_uses_given_split_and_batch_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'a.jpg'
            image.write_bytes(b'jpg-bytes')
            cfg = dict(api_key='secretkey', project='proj')
            opener = FakeOpener([dict(success=True, id='id-a'), dict(success=True)])
            rd.upload_with_retry(cfg, image, '<annotation/>', 'valid', 'universe:kicksquad/scooter-detect',
                                  opener=opener)
            self.assertIn('split=valid', opener.calls[0])
            self.assertIn('batch=universe%3Akicksquad/scooter-detect', opener.calls[0])


class NoSecretsPrintedTests(unittest.TestCase):
    def test_format_counts_and_upload_error_never_contain_a_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'a.jpg'
            image.write_bytes(b'jpg-bytes')
            cfg = dict(api_key='very-secret-key', project='proj')
            body = f'oops very-secret-key leaked'.encode('utf-8')
            opener = FakeOpener([http_error(400, body)])
            try:
                rd.upload_with_retry(cfg, image, '<annotation/>', 'train', 'universe:ws/proj', opener=opener)
            except rs.RoboflowError as err:
                self.assertNotIn('very-secret-key', str(err))
            counts = {name: 1 for name in TARGET}
            self.assertNotIn('very-secret-key', rd.format_counts(counts))


if __name__ == '__main__':
    unittest.main()


def test_voc_xml_trims_boxes_to_the_image():
    from utils import roboflow_sync
    xml = roboflow_sync.voc_xml_from_yolo_lines(
        ['0 0.95 0.5 0.3 0.4', '0 1.5 0.5 0.2 0.2'], ['person'], 'x.jpg', 100, 100)
    assert '<xmax>100</xmax>' in xml
    assert xml.count('<object>') == 1   # the box wholly outside is dropped


class ListLike(list):
    """Stands in for a torch tensor's .tolist() in fake ultralytics results."""
    def tolist(self):
        return list(self)


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = ListLike(xyxy)
        self.cls = ListLike(cls)
        self.conf = ListLike(conf)

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _make_image(path, size=(64, 48)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size).save(path)


# ------------------------------------------------------------- local sources

class LocalSourceImagesTests(unittest.TestCase):
    def test_ignores_holdout_and_splits_by_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp) / 'selected'
            for split in ('train', 'valid', 'holdout'):
                _make_image(dirpath / split / 'images' / f'own-cam-{split}.jpg')
            pairs = rd.local_source_images(dirpath)
            got = sorted((p.name, split) for p, split in pairs)
            self.assertEqual(got, [('own-cam-train.jpg', 'train'), ('own-cam-valid.jpg', 'valid')])

    def test_missing_split_dirs_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp) / 'selected'
            _make_image(dirpath / 'train' / 'images' / 'own-cam-a.jpg')
            pairs = rd.local_source_images(dirpath)
            self.assertEqual([p.name for p, _s in pairs], ['own-cam-a.jpg'])


class PrepareLocalTests(unittest.TestCase):
    def _fake_teacher_predict_batches(self, teacher, paths, batch=16, conf=0.5, device='mps'):
        self.recorded_conf = conf
        for _p in paths:
            yield FakeResult(FakeBoxes([[0, 0, 10, 10]], [0], [0.9]))

    def setUp(self):
        self._orig = rd.tfc.teacher_predict_batches
        rd.tfc.teacher_predict_batches = self._fake_teacher_predict_batches
        self.recorded_conf = None

    def tearDown(self):
        rd.tfc.teacher_predict_batches = self._orig

    def test_batch_name_split_and_filename_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp) / 'selected'
            _make_image(dirpath / 'train' / 'images' / 'own-cam-1.jpg')
            _make_image(dirpath / 'valid' / 'images' / 'own-cam-2.jpg')
            _make_image(dirpath / 'holdout' / 'images' / 'own-cam-3.jpg')  # must be ignored

            teacher_index_map = {0: TARGET.index('person')}
            records, counts = rd.prepare_local(dirpath, teacher=object(),
                                                teacher_index_map=teacher_index_map, teacher_conf=0.35)

            self.assertEqual(len(records), 2)
            by_name = {r['orig_name']: r for r in records}
            self.assertEqual(set(by_name), {'own-cam-1.jpg', 'own-cam-2.jpg'})
            self.assertEqual(by_name['own-cam-1.jpg']['split'], 'train')
            self.assertEqual(by_name['own-cam-2.jpg']['split'], 'valid')
            for r in records:
                self.assertEqual(r['final_name'], r['orig_name'])   # already unique, passed through as-is
                self.assertEqual(r['batch'], 'own:selected')
            self.assertEqual(counts['person'], 2)   # teacher labelled every image
            self.assertEqual(self.recorded_conf, 0.35)

    def test_no_images_no_teacher_calls_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp) / 'empty'
            records, counts = rd.prepare_local(dirpath, teacher=object(), teacher_index_map={}, teacher_conf=0.5)
            self.assertEqual(records, [])
            self.assertEqual(sum(counts.values()), 0)


class ExtraLabelsTests(unittest.TestCase):
    """Hand-checked boxes (e.g. pushchairs, which the COCO teacher can't label)
    are added to local frames and replace a teacher box of another class on
    the same object."""

    def setUp(self):
        self._orig = rd.tfc.teacher_predict_batches
        # teacher: a person at the left, and the pushchair (right) called a bicycle
        rd.tfc.teacher_predict_batches = lambda teacher, paths, batch=16, conf=0.5, device='mps': (
            FakeResult(FakeBoxes([[0, 0, 20, 40], [34, 10, 60, 44]], [0, 1], [0.9, 0.6])) for _ in paths)

    def tearDown(self):
        rd.tfc.teacher_predict_batches = self._orig

    def test_extra_box_added_and_overlapping_teacher_box_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp) / 'selected'
            _make_image(dirpath / 'train' / 'images' / 'own-cam-20260930-100000.jpg')   # 64x48
            _make_image(dirpath / 'train' / 'images' / 'own-cam-20260930-110000.jpg')
            extra = {'own-cam-20260930-100000.jpg': [['stroller', 34 / 64, 10 / 48, 60 / 64, 44 / 48]]}
            index_map = {0: TARGET.index('person'), 1: TARGET.index('bicycle')}
            records, counts = rd.prepare_local(dirpath, teacher=object(), teacher_index_map=index_map,
                                               extra_labels=extra)
            by_name = {r['orig_name']: r for r in records}
            classes = lambda n: sorted(TARGET[int(l.split()[0])] for l in by_name[n]['lines'])
            self.assertEqual(classes('own-cam-20260930-100000.jpg'), ['person', 'stroller'])
            self.assertEqual(classes('own-cam-20260930-110000.jpg'), ['bicycle', 'person'])   # untouched
            self.assertEqual(counts['stroller'], 1)

    def test_unknown_class_in_extra_labels_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp) / 'selected'
            _make_image(dirpath / 'train' / 'images' / 'own-cam-20260930-100000.jpg')
            with self.assertRaises(ValueError):
                rd.prepare_local(dirpath, teacher=object(), teacher_index_map={},
                                 extra_labels={'own-cam-20260930-100000.jpg': [['pram', 0, 0, .5, .5]]})


class MainArgValidationTests(unittest.TestCase):
    """main()'s early --source/--local and --teacher validation, before any
    config file, network, or model loading is touched."""

    def _run_main_with_argv(self, argv):
        old_argv = sys.argv
        sys.argv = argv
        try:
            with self.assertRaises(SystemExit) as ctx:
                rd.main()
            return str(ctx.exception)
        finally:
            sys.argv = old_argv

    def test_requires_source_or_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            msg = self._run_main_with_argv(['roboflow_dataset.py', '--data', tmp])
            self.assertIn('--source', msg)
            self.assertIn('--local', msg)

    def test_local_works_with_zero_source_args_past_validation(self):
        # teacher-none-with-local should fire, proving --local alone (no
        # --source) got past the "need at least one" check.
        with tempfile.TemporaryDirectory() as tmp:
            msg = self._run_main_with_argv(
                ['roboflow_dataset.py', '--data', tmp, '--local', str(Path(tmp) / 'selected'), '--teacher', 'none'])
            self.assertIn('--teacher none', msg)

    def test_teacher_none_with_local_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            msg = self._run_main_with_argv(
                ['roboflow_dataset.py', '--data', tmp, '--local', str(Path(tmp) / 'selected'), '--teacher', 'none'])
            self.assertIn('--teacher none', msg)
            self.assertIn('--local', msg)

    def test_teacher_none_without_local_is_fine_past_this_check(self):
        # sources-only + --teacher none should NOT hit the local/teacher
        # error; it fails later (no api key configured) instead.
        with tempfile.TemporaryDirectory() as tmp:
            msg = self._run_main_with_argv(
                ['roboflow_dataset.py', '--data', tmp, '--source', 'ws/proj', '--teacher', 'none'])
            self.assertNotIn('cannot be used with --local', msg)


def test_polygon_label_lines_become_enclosing_boxes():
    from utils import roboflow_sync
    poly = '0 0.2 0.5 0.4 0.5 0.4 0.9 0.2 0.9'
    assert roboflow_sync.yolo_box_line(poly, {0: 7}) == '7 0.300000 0.700000 0.200000 0.400000'
    assert roboflow_sync.yolo_box_line('0 0.5 0.5 0.2 0.2') == '0 0.5 0.5 0.2 0.2'
    assert roboflow_sync.yolo_box_line('3 0.5 0.5 0.2 0.2', {0: 1}) is None
    assert roboflow_sync.yolo_box_line('0 0.1 0.2 0.3') is None
