import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.coreml_yolo import build_canonical, remap_class_ids

COCO = [l.strip() for l in (ROOT / 'models' / 'coco.names').read_text().splitlines() if l.strip()]


class BuildCanonicalTests(unittest.TestCase):
    def test_stock_coco_list_is_identity(self):
        canonical, index_map = build_canonical(COCO, COCO)
        # Canonical always reserves slots for the fixed extras (80-82), but a
        # model that only reports the 80 COCO names maps identically onto them.
        self.assertEqual(canonical[:80], COCO)
        self.assertEqual(canonical[80:83], ['stroller', 'child', 'scooter'])
        for i in range(len(COCO)):
            self.assertEqual(index_map[i], i)

    def test_eight_class_alphabetical_roboflow_list(self):
        model_names = ['bicycle', 'car', 'child', 'dog', 'person', 'scooter', 'stroller', 'truck']
        canonical, index_map = build_canonical(model_names, COCO)
        self.assertEqual(canonical[:80], COCO)
        self.assertEqual(canonical[80:83], ['stroller', 'child', 'scooter'])

        def idx(name):
            return model_names.index(name)

        self.assertEqual(index_map[idx('person')], COCO.index('person'))
        self.assertEqual(index_map[idx('car')], COCO.index('car'))
        self.assertEqual(index_map[idx('bicycle')], COCO.index('bicycle'))
        self.assertEqual(index_map[idx('dog')], COCO.index('dog'))
        self.assertEqual(index_map[idx('truck')], COCO.index('truck'))
        self.assertEqual(index_map[idx('stroller')], 80)
        self.assertEqual(index_map[idx('child')], 81)
        self.assertEqual(index_map[idx('scooter')], 82)

    def test_aliases_map_onto_canonical_names(self):
        model_names = ['human', 'Kids', 'e-scooter']
        canonical, index_map = build_canonical(model_names, COCO)
        self.assertEqual(index_map[0], COCO.index('person'))
        self.assertEqual(index_map[1], canonical.index('child'))
        self.assertEqual(index_map[2], canonical.index('scooter'))

    def test_unknown_extra_appended_after_fixed_extras(self):
        model_names = ['person', 'wheelie bin']
        canonical, index_map = build_canonical(model_names, COCO)
        # Fixed extras always occupy 80, 81, 82 even if unused by this model.
        self.assertEqual(canonical[80:83], ['stroller', 'child', 'scooter'])
        self.assertEqual(canonical[83], 'wheelie bin')
        self.assertEqual(index_map[0], COCO.index('person'))
        self.assertEqual(index_map[1], 83)


class VehicleAssistCanonicalIdsTests(unittest.TestCase):
    """The vehicle-assist model's own 8-class vocabulary must land on the same
    canonical ids as the primary model (car=2, truck=7, stroller=80), since
    utils.assist_merge.merge_assist filters by those ids across both."""

    def test_assist_classes_map_onto_primary_canonical_ids(self):
        assist_names = ['bicycle', 'car', 'child', 'dog', 'person', 'scooter', 'stroller', 'truck']
        canonical, index_map = build_canonical(assist_names, COCO)
        self.assertEqual(canonical[:83], COCO + ['stroller', 'child', 'scooter'])
        self.assertEqual(index_map[assist_names.index('car')], COCO.index('car'))
        self.assertEqual(index_map[assist_names.index('truck')], COCO.index('truck'))
        self.assertEqual(index_map[assist_names.index('stroller')], 80)
        self.assertEqual(COCO.index('car'), 2)
        self.assertEqual(COCO.index('truck'), 7)


class ApplyModelClassNamesStrollerTests(unittest.TestCase):
    """apply_model_class_names (clearcam.py) mirrors a model's canonical_names
    onto class_labels; when a model (assist or -home) has 'stroller' in its
    own vocabulary, canonical_names always includes it at id 80 - so
    class_labels ends up with 'stroller' too."""

    class FakeModel:
        def __init__(self, names, canonical_names=None):
            self.names = names
            self.canonical_names = canonical_names

    @staticmethod
    def apply_model_class_names(model, class_labels, color_dict):
        # Mirrors clearcam.apply_model_class_names without importing clearcam.py
        # (which has heavy hardware-only top-level imports).
        names = getattr(model, 'canonical_names', None) or getattr(model, 'names', None)
        if not names:
            return
        class_labels[:] = names
        color_dict.clear()
        color_dict.update({label: (0, 0, 0) for label in class_labels})

    def test_stroller_present_when_assist_model_canonical_names_used(self):
        assist_names = ['bicycle', 'car', 'child', 'dog', 'person', 'scooter', 'stroller', 'truck']
        canonical, _ = build_canonical(assist_names, COCO)
        model = self.FakeModel(names=assist_names, canonical_names=canonical)
        class_labels = list(COCO)
        color_dict = {}
        self.apply_model_class_names(model, class_labels, color_dict)
        self.assertIn('stroller', class_labels)
        self.assertEqual(class_labels.index('stroller'), 80)

    def test_stock_coco_model_canonical_names_also_includes_stroller(self):
        # build_canonical always reserves 80-82 for the fixed extras, even for
        # a plain COCO model, so the primary detector's class_labels already
        # carry 'stroller' regardless of whether assist is installed.
        canonical, _ = build_canonical(COCO, COCO)
        model = self.FakeModel(names=COCO, canonical_names=canonical)
        class_labels = list(COCO)
        color_dict = {}
        self.apply_model_class_names(model, class_labels, color_dict)
        self.assertIn('stroller', class_labels)


class RemapClassIdsTests(unittest.TestCase):
    def test_drops_unmapped_classes_and_remaps_others(self):
        # model has 3 classes: 0 -> canonical 5, 1 -> unmapped (-1), 2 -> canonical 2
        lookup = np.array([5, -1, 2], dtype=np.int64)
        preds = np.array([
            [0, 0, 10, 10, 0.9, 0],
            [0, 0, 10, 10, 0.8, 1],
            [0, 0, 10, 10, 0.7, 2],
        ], dtype=np.float32)
        out = remap_class_ids(preds, lookup)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(sorted(out[:, 5].astype(int).tolist()), [2, 5])

    def test_empty_preds_passthrough(self):
        lookup = np.array([0], dtype=np.int64)
        preds = np.zeros((0, 6), dtype=np.float32)
        out = remap_class_ids(preds, lookup)
        self.assertEqual(out.shape, (0, 6))


if __name__ == '__main__':
    unittest.main()
