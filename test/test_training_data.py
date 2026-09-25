import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import roboflow_sync as rs
from script import train_from_corrections as tfc

COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'fire hydrant', 'stop sign', 'cat', 'dog']
MERGED = COCO + ['stroller', 'child', 'scooter']  # the fixed vocabulary once new-classes are appended


class NormaliseClassTests(unittest.TestCase):
    def test_parcel_is_dropped_not_aliased(self):
        # 'package'/'parcel' are no longer in the vocabulary; normalise_class
        # just lowercases them since there's nothing in MERGED to match.
        self.assertEqual(rs.normalise_class('Parcel', MERGED), 'parcel')
        self.assertNotIn(rs.normalise_class('Parcel', MERGED), MERGED)

    def test_coco_names_keep_case_from_base(self):
        self.assertEqual(rs.normalise_class('Person', COCO), 'person')
        self.assertEqual(rs.normalise_class('Dog', COCO), 'dog')

    def test_person_aliases(self):
        for raw in ('human', 'pedestrian', 'person_cane', 'person cane', 'People', 'Persons', 'Adult'):
            self.assertEqual(rs.normalise_class(raw, MERGED), 'person', raw)

    def test_child_aliases(self):
        for raw in ('kid', 'kids', 'baby', 'Children'):
            self.assertEqual(rs.normalise_class(raw, MERGED), 'child', raw)

    def test_scooter_and_stroller_aliases(self):
        self.assertEqual(rs.normalise_class('e-scooter', MERGED), 'scooter')
        self.assertEqual(rs.normalise_class('kickscooter', MERGED), 'scooter')
        self.assertEqual(rs.normalise_class('pram', MERGED), 'stroller')
        self.assertEqual(rs.normalise_class('baby carriage', MERGED), 'stroller')

    def test_ambulance_and_bike_aliases(self):
        self.assertEqual(rs.normalise_class('ambulance', MERGED), 'truck')
        self.assertEqual(rs.normalise_class('bike', MERGED), 'bicycle')
        self.assertEqual(rs.normalise_class('motorbike', MERGED), 'motorcycle')

    def test_stop_alias(self):
        self.assertEqual(rs.normalise_class('stop', MERGED), 'stop sign')

    def test_underscore_to_space_matches_coco_without_an_alias(self):
        self.assertEqual(rs.normalise_class('fire_hydrant', MERGED), 'fire hydrant')

    def test_numeric_prefix_is_stripped(self):
        self.assertEqual(rs.normalise_class('3-stroller', MERGED), 'stroller')
        self.assertEqual(rs.normalise_class('0-human', MERGED), 'person')

    def test_unknown_name_lowercased_but_not_in_vocabulary(self):
        result = rs.normalise_class('Scooter X', MERGED)
        self.assertEqual(result, 'scooter x')
        self.assertNotIn(result, [n.lower() for n in MERGED])

    def test_strip_and_lowercase(self):
        self.assertEqual(rs.normalise_class('  Car  ', COCO), 'car')


class BuildIndexMapTests(unittest.TestCase):
    def test_many_to_one_mapping(self):
        src_names = ['human', 'pedestrian', 'Person', 'Dog']
        index_map = rs.build_index_map(src_names, MERGED)
        person_idx = MERGED.index('person')
        dog_idx = MERGED.index('dog')
        self.assertEqual(index_map[0], person_idx)
        self.assertEqual(index_map[1], person_idx)
        self.assertEqual(index_map[2], person_idx)
        self.assertEqual(index_map[3], dog_idx)

    def test_dropped_class_not_in_merged_is_omitted(self):
        merged = ['person', 'car']  # 'raccoon' is not in the vocabulary
        index_map = rs.build_index_map(['person', 'raccoon', 'car'], merged)
        self.assertEqual(index_map, {0: 0, 2: 1})

    def test_parcel_is_omitted_now_that_package_is_gone(self):
        index_map = rs.build_index_map(['Parcel', 'Person'], MERGED)
        self.assertNotIn(0, index_map)
        self.assertEqual(index_map[1], MERGED.index('person'))


class BuildIndexMapWithDropsTests(unittest.TestCase):
    def test_drop_classes_applied_before_matching(self):
        # 'dog' would normally map into MERGED, but --drop-classes excludes it.
        index_map = tfc.build_index_map_with_drops(['dog', 'person'], MERGED, drop_classes=['dog'])
        self.assertNotIn(0, index_map)
        self.assertEqual(index_map[1], MERGED.index('person'))

    def test_no_drops_matches_build_index_map(self):
        plain = rs.build_index_map(['dog', 'person'], MERGED)
        with_drops = tfc.build_index_map_with_drops(['dog', 'person'], MERGED, drop_classes=[])
        self.assertEqual(plain, with_drops)


class AugmentLabelsTests(unittest.TestCase):
    def test_adds_box_for_unlabelled_coco_class(self):
        # existing_lines label only class 5 (in a 100x100 image); teacher adds a car (class 2)
        existing = ['5 0.5 0.5 0.2 0.2']
        teacher_boxes = [(2, [10, 10, 30, 30], 0.9)]
        out = tfc.augment_labels(existing, teacher_boxes, labelled_classes={5}, width=100, height=100)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[1].startswith('2 '))

    def test_skips_labelled_class(self):
        existing = ['5 0.5 0.5 0.2 0.2']
        teacher_boxes = [(5, [10, 10, 30, 30], 0.9)]
        out = tfc.augment_labels(existing, teacher_boxes, labelled_classes={5}, width=100, height=100)
        self.assertEqual(out, existing)

    def test_skips_overlapping_box(self):
        # existing box covers [40,40,60,60] in a 100x100 image (cx=.5,cy=.5,w=.2,h=.2)
        existing = ['5 0.5 0.5 0.2 0.2']
        teacher_boxes = [(2, [41, 41, 61, 61], 0.9)]  # heavily overlaps the existing box
        out = tfc.augment_labels(existing, teacher_boxes, labelled_classes=set(), width=100, height=100)
        self.assertEqual(out, existing)

    def test_keeps_originals(self):
        existing = ['5 0.5 0.5 0.2 0.2', '1 0.1 0.1 0.05 0.05']
        out = tfc.augment_labels(existing, [], labelled_classes=set(), width=100, height=100)
        self.assertEqual(out, existing)
        self.assertIsNot(out, existing)  # never mutates the input list

    def test_skips_teacher_person_overlapping_child(self):
        # class 20 = child, covering [0,0,50,50] in a 100x100 image. A teacher
        # 'person' (class 0) box at [15,15,65,65] overlaps it at IoU ~= 0.32 -
        # above the 0.3 person-vs-child threshold, but below the generic 0.5
        # overlap threshold, so only the new rule catches it.
        existing = ['20 0.25 0.25 0.5 0.5']
        teacher_boxes = [(0, [15, 15, 65, 65], 0.9)]
        out = tfc.augment_labels(existing, teacher_boxes, labelled_classes=set(), width=100, height=100,
                                  person_class=0, child_class=20)
        self.assertEqual(out, existing)

    def test_teacher_person_far_from_child_is_kept(self):
        existing = ['20 0.1 0.1 0.1 0.1']  # child in a corner
        teacher_boxes = [(0, [60, 60, 80, 80], 0.9)]  # unrelated person elsewhere
        out = tfc.augment_labels(existing, teacher_boxes, labelled_classes=set(), width=100, height=100,
                                  person_class=0, child_class=20)
        self.assertEqual(len(out), 2)


class SpecParsingTests(unittest.TestCase):
    def test_project_and_version_only(self):
        self.assertEqual(tfc.parse_roboflow_spec('sideguide:1', 'scottsdale'), ('scottsdale', 'sideguide', 1))

    def test_workspace_project_version(self):
        self.assertEqual(tfc.parse_roboflow_spec('other-ws/clearcam-home:2', 'my-ws'), ('other-ws', 'clearcam-home', 2))

    def test_missing_colon_raises(self):
        with self.assertRaises(ValueError):
            tfc.parse_roboflow_spec('sideguide', 'my-ws')

    def test_non_integer_version_raises(self):
        with self.assertRaises(ValueError):
            tfc.parse_roboflow_spec('sideguide:v1', 'my-ws')

    def test_no_default_workspace_raises(self):
        with self.assertRaises(ValueError):
            tfc.parse_roboflow_spec('sideguide:1', '')


class BuildMergedNamesTests(unittest.TestCase):
    def test_appends_new_classes_in_given_order(self):
        base = ['person', 'car']
        merged = tfc.build_merged_names(base, ['stroller', 'child', 'scooter'])
        self.assertEqual(merged, ['person', 'car', 'stroller', 'child', 'scooter'])

    def test_fixed_regardless_of_any_dataset_content(self):
        # build_merged_names no longer looks at dataset class lists at all -
        # the vocabulary is base COCO + --new-classes, full stop.
        base = ['person', 'car']
        merged = tfc.build_merged_names(base, ['stroller', 'child', 'scooter'])
        self.assertEqual(merged, tfc.build_merged_names(base, ['stroller', 'child', 'scooter']))

    def test_skips_blank_and_duplicate_entries(self):
        base = ['person']
        merged = tfc.build_merged_names(base, ['child', '', 'person', 'child'])
        self.assertEqual(merged, ['person', 'child'])

    def test_empty_new_classes_keeps_base_only(self):
        self.assertEqual(tfc.build_merged_names(['person', 'car'], []), ['person', 'car'])


class BalanceRepeatFactorTests(unittest.TestCase):
    def test_typical_ratio(self):
        # 100 local images against 3300 external: local*f / (local*f + external) ~= .10
        # f = .10*3300 / (100*.90) = 330/90 = 3.67 -> rounds to 4
        factor = tfc.balance_repeat_factor(100, 3300)
        self.assertEqual(factor, 4)

    def test_clamped_to_min_factor(self):
        # Already well above 10%: solved factor would be < 1
        factor = tfc.balance_repeat_factor(1000, 100)
        self.assertEqual(factor, 1)

    def test_clamped_to_max_factor(self):
        factor = tfc.balance_repeat_factor(1, 100000)
        self.assertEqual(factor, 20)

    def test_no_external_data_defaults_to_min(self):
        self.assertEqual(tfc.balance_repeat_factor(100, 0), 1)

    def test_no_local_data_defaults_to_min(self):
        self.assertEqual(tfc.balance_repeat_factor(0, 100), 1)


class HoldoutFilenamesTests(unittest.TestCase):
    def test_fraction_and_determinism(self):
        names = [f'img{i}.jpg' for i in range(100)]
        h1 = tfc.holdout_filenames(names, fraction=0.10)
        h2 = tfc.holdout_filenames(list(reversed(names)), fraction=0.10)
        self.assertEqual(len(h1), 10)
        self.assertEqual(h1, h2)  # independent of input order

    def test_at_least_one_for_small_lists(self):
        self.assertEqual(len(tfc.holdout_filenames(['a.jpg', 'b.jpg'], fraction=0.10)), 1)

    def test_empty_input(self):
        self.assertEqual(tfc.holdout_filenames([]), set())


class NothingNewToLearnTests(unittest.TestCase):
    def test_new_class_proceeds(self):
        self.assertFalse(tfc.nothing_new_to_learn(['stroller'], disagreements=0))

    def test_disagreement_proceeds(self):
        self.assertFalse(tfc.nothing_new_to_learn([], disagreements=1))

    def test_neither_refuses(self):
        self.assertTrue(tfc.nothing_new_to_learn([], disagreements=0))

    def test_both_proceeds(self):
        self.assertFalse(tfc.nothing_new_to_learn(['child'], disagreements=3))


if __name__ == '__main__':
    unittest.main()
