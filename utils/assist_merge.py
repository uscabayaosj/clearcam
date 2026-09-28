"""Merge a vehicle-assist detector's predictions into the primary detector's.

The primary detector (stock yolo11s, COCO classes) covers everything except
cars and trucks, where it undercounts badly on the owner's cameras. A second
"vehicle assist" model - a Roboflow-trained YOLO11n specialised on
[bicycle, car, child, dog, person, scooter, stroller, truck] - is far more
reliable for cars/trucks/strollers but much worse for people/bikes/dogs, so
only those few classes are taken from it.

Both prediction arrays are expected in the *same* canonical class-id space
(see detection.coreml_yolo.build_canonical, which both models' CoreMLYolo
wrappers already run their raw class ids through) and the same frame-pixel
coordinates, as Nx6 arrays of [x1, y1, x2, y2, score, class_id].
"""
import numpy as np

# car, truck, stroller (canonical ids: COCO order + the fixed extras).
DEFAULT_TAKE_IDS = frozenset({2, 7, 80})
DEFAULT_MIN_CONFIDENCE = 0.4


def merge_assist(primary_preds, assist_preds, take_ids=DEFAULT_TAKE_IDS,
                  min_confidence=DEFAULT_MIN_CONFIDENCE):
    """Return primary_preds with `take_ids` classes replaced by assist_preds.

    - Rows in primary_preds whose class is in take_ids are dropped (the
      assist model owns those classes now).
    - Rows in assist_preds whose class is in take_ids and whose score is >=
      min_confidence are appended.
    - Everything else is untouched. Output keeps the Nx6 shape/dtype of
      primary_preds (falls back to float32 if primary_preds is empty and
      untyped).
    """
    primary_preds = np.asarray(primary_preds)
    assist_preds = np.asarray(assist_preds)
    dtype = primary_preds.dtype if primary_preds.dtype.kind == 'f' else np.float32

    take_ids_arr = np.array(sorted(take_ids), dtype=np.int64) if take_ids else np.zeros(0, dtype=np.int64)

    if primary_preds.shape[0]:
        primary_classes = primary_preds[:, 5].astype(np.int64)
        keep = ~np.isin(primary_classes, take_ids_arr)
        kept_primary = primary_preds[keep]
    else:
        kept_primary = primary_preds.reshape(0, 6)

    if assist_preds.shape[0]:
        assist_classes = assist_preds[:, 5].astype(np.int64)
        take = np.isin(assist_classes, take_ids_arr) & (assist_preds[:, 4] >= min_confidence)
        taken_assist = assist_preds[take]
    else:
        taken_assist = assist_preds.reshape(0, 6)

    merged = np.concatenate([kept_primary, taken_assist], axis=0) if (
        kept_primary.shape[0] or taken_assist.shape[0]
    ) else np.zeros((0, 6), dtype=dtype)
    return merged.astype(dtype)
