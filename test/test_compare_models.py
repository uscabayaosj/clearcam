import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import compare_models as cm


# ---------------------------------------------------------------------- IoU

def test_iou_perfect_overlap():
    assert cm.iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap():
    assert cm.iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=200-50=150
    assert abs(cm.iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_iou_degenerate_box_is_zero():
    assert cm.iou_xyxy((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------- matching

def test_match_count_matches_within_threshold():
    dets = [((0, 0, 10, 10), 0.9)]
    refs = [(0, 0, 10, 10)]
    assert cm.match_count(dets, refs) == 1


def test_match_count_below_iou_threshold_unmatched():
    dets = [((0, 0, 10, 10), 0.9)]
    refs = [(20, 20, 30, 30)]
    assert cm.match_count(dets, refs) == 0


def test_match_count_greedy_by_confidence_one_ref_only_matched_once():
    dets = [((0, 0, 10, 10), 0.5), ((1, 1, 11, 11), 0.95)]
    refs = [(0, 0, 10, 10)]
    assert cm.match_count(dets, refs) == 1


def test_match_count_two_refs_two_dets():
    dets = [((0, 0, 10, 10), 0.9), ((100, 100, 110, 110), 0.8)]
    refs = [(0, 0, 10, 10), (100, 100, 110, 110)]
    assert cm.match_count(dets, refs) == 2


def test_match_count_no_detections_or_refs():
    assert cm.match_count([], []) == 0
    assert cm.match_count([], [(0, 0, 10, 10)]) == 0
    assert cm.match_count([((0, 0, 10, 10), 0.9)], []) == 0


# ------------------------------------------------------ counts / recall / precision

def test_class_counts_single_image():
    dets = {'person': [((0, 0, 10, 10), 0.9)], 'car': []}
    refs = {'person': [(0, 0, 10, 10)], 'car': [(5, 5, 15, 15)]}
    counts = cm.class_counts(dets, refs, ['person', 'car'])
    assert counts['person'] == dict(det=1, ref=1, matched=1)
    assert counts['car'] == dict(det=0, ref=1, matched=0)


def test_aggregate_sums_across_frames():
    frame_a = dict(person=dict(det=1, ref=1, matched=1), car=dict(det=0, ref=1, matched=0))
    frame_b = dict(person=dict(det=1, ref=0, matched=0), car=dict(det=2, ref=1, matched=1))
    totals = cm.aggregate([frame_a, frame_b], ['person', 'car'])
    assert totals['person'] == dict(det=2, ref=1, matched=1)
    assert totals['car'] == dict(det=2, ref=2, matched=1)


def test_recall_precision_basic():
    totals = dict(person=dict(det=2, ref=2, matched=1))
    rp = cm.recall_precision(totals)
    assert rp['person']['recall'] == 0.5
    assert rp['person']['precision'] == 0.5


def test_recall_precision_undefined_when_no_reference_or_no_detections():
    totals = dict(a=dict(det=0, ref=0, matched=0))
    rp = cm.recall_precision(totals)
    assert rp['a']['recall'] is None
    assert rp['a']['precision'] is None


def test_recall_zero_when_reference_present_but_nothing_matched():
    totals = dict(a=dict(det=1, ref=3, matched=0))
    rp = cm.recall_precision(totals)
    assert rp['a']['recall'] == 0.0
    assert rp['a']['precision'] == 0.0


# --------------------------------------------------------------------- verdict

def test_verdict_worse_when_person_recall_drops_more_than_threshold():
    stock = {'person': {'recall': 0.8}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.65}, 'car': {'recall': 0.75}}
    assert cm.verdict(stock, candidate) == 'worse'


def test_verdict_worse_when_car_recall_drops_more_than_threshold():
    stock = {'person': {'recall': 0.8}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.9}, 'car': {'recall': 0.55}}
    assert cm.verdict(stock, candidate) == 'worse'


def test_verdict_better_when_ahead_on_both_classes():
    stock = {'person': {'recall': 0.8}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.85}, 'car': {'recall': 0.75}}
    assert cm.verdict(stock, candidate) == 'better'


def test_verdict_mixed_on_small_tradeoff_within_threshold():
    stock = {'person': {'recall': 0.8}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.85}, 'car': {'recall': 0.68}}
    assert cm.verdict(stock, candidate) == 'mixed'


def test_verdict_mixed_when_identical():
    stock = {'person': {'recall': 0.8}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.8}, 'car': {'recall': 0.7}}
    assert cm.verdict(stock, candidate) == 'mixed'


def test_verdict_just_under_threshold_is_not_worse():
    stock = {'person': {'recall': 0.80}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.71}, 'car': {'recall': 0.7}}  # 0.09 below: not "more than" 0.10
    assert cm.verdict(stock, candidate) != 'worse'


def test_verdict_treats_missing_recall_as_zero():
    stock = {'person': {'recall': None}, 'car': {'recall': 0.7}}
    candidate = {'person': {'recall': 0.0}, 'car': {'recall': 0.75}}
    assert cm.verdict(stock, candidate) == 'better'


# ------------------------------------------------------------ class-name mapping

def test_restrict_names_case_insensitive_and_drops_unmapped():
    names = {0: 'Person', 1: 'bicycle', 2: 'traffic light'}
    out = cm.restrict_names(names, cm.TARGET_CLASSES)
    assert out == {0: 'person', 1: 'bicycle'}


def test_boxes_by_class_groups_and_drops_unmapped():
    idx_to_name = {0: 'person', 1: 'car'}
    xyxy = [[0, 0, 10, 10], [1, 1, 2, 2], [5, 5, 15, 15]]
    cls = [0, 99, 1]
    conf = [0.9, 0.5, 0.8]
    out = cm.boxes_by_class(xyxy, cls, conf, idx_to_name)
    assert set(out) == {'person', 'car'}
    assert out['person'] == [((0, 0, 10, 10), 0.9)]
    assert out['car'] == [((5, 5, 15, 15), 0.8)]
