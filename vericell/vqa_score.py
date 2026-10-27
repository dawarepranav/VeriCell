"""MorphoOrgaVQA scoring, re-implemented from the benchmark's published protocol (MorphoOrgaVQA/README.md).

Per numeric field: absolute percentage error |pred - gt| / |gt| * 100 (absolute error when |gt| < 1e-6).
Credit: 1.0 if APE <= 15%, 0.5 if <= 30%, else 0; missing or non-finite answers get 0 credit.
Reported as median APE per (category, slot type, question mode), as in the paper's Table 1.
"""

from __future__ import annotations

import math

import numpy as np

from .measure import answer, measure_label_mask

CATEGORY_METRIC = {"Area": "Area_outer", "Perimeter": "Perimeter_outer", "Roundness": "roundness",
                   "Roughness": "perimeter_diff"}
FULL, HALF = 15.0, 30.0
ABS_FALLBACK, ABS_FULL = 1e-6, 1e-3


def groundtruth(labels: np.ndarray) -> dict:
    """Answers for every benchmark slot, from an instance-label mask (same as the official generator)."""
    table = measure_label_mask(labels)
    out = {}
    for c, m in CATEGORY_METRIC.items():
        out[f"{c}__max"] = answer(table, m, "max", ["Area_outer", "x", "y"] if c == "Area" else [m])
        out[f"{c}__tertile_mean"] = answer(table, m, "tertile_mean", [m])
    return out


def score_field(pred, gt: float) -> dict:
    if pred is None or not isinstance(pred, (int, float)) or not math.isfinite(pred):
        return {"pct_error": None, "credit": 0.0, "status": "missing"}
    if abs(gt) < ABS_FALLBACK:
        err = abs(pred - gt)
        return {"pct_error": None, "credit": 1.0 if err <= ABS_FULL else 0.5 if err <= 3 * ABS_FULL else 0.0,
                "status": "abs"}
    pct = abs(pred - gt) / abs(gt) * 100
    return {"pct_error": pct, "credit": 1.0 if pct <= FULL else 0.5 if pct <= HALF else 0.0, "status": "ok"}


def score(predictions: dict, gts: dict) -> list[dict]:
    """predictions[image][slot][mode] = {"final_answer": answer dict} (the benchmark's prediction format);
    gts[image][slot] = answer dict. Returns per-field records."""
    records = []
    for image, slots in gts.items():
        for slot, gt in slots.items():
            if not gt:
                continue
            category, slot_type = slot.split("__")
            for mode in ("clear", "open"):
                pred = ((predictions.get(image, {}).get(slot, {}).get(mode)) or {}).get("final_answer") or {}
                for field, g in gt.items():
                    if g is None:
                        continue
                    records.append({"image": image, "category": category, "slot_type": slot_type, "mode": mode,
                                    "field": field, "gt": g, "pred": pred.get(field), **score_field(pred.get(field), g)})
    return records
