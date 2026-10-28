"""Deterministic measurement engine. The LLM plans; this code does all the arithmetic.

Per-object metrics follow MorphoOrgaVQA's definitions (MorphoOrgaAgent tools/morphology_tools.py)
exactly, so answers are comparable with its ground truth:
  Area_outer       cv2.contourArea of the object's outer contour (RETR_CCOMP, first parentless)
  Perimeter_outer  cv2.arcLength of that contour
  x, y             centroid from the contour's moments
  roundness        4*pi*A / P^2
  perimeter_diff   P - 2*sqrt(pi*A)   (a.k.a. roughness: boundary excess over a same-area circle)
"""

from __future__ import annotations

import cv2
import numpy as np

METRICS = ["Area_outer", "Perimeter_outer", "roundness", "perimeter_diff"]


def outer_contour(obj: np.ndarray):
    contours, hierarchy = cv2.findContours(obj.astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if hierarchy is None or not contours:
        return None
    for c, h in zip(contours, hierarchy[0]):
        if h[3] == -1:
            return c
    return contours[0]


def measure_object(obj: np.ndarray) -> dict | None:
    """Metrics for one boolean object mask (cropped or full-size)."""
    c = outer_contour(obj)
    if c is None:
        return None
    area, perim = float(cv2.contourArea(c)), float(cv2.arcLength(c, True))
    m = cv2.moments(c)
    x, y = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (0.0, 0.0)
    return {
        "Area_outer": area,
        "Perimeter_outer": perim,
        "x": float(x),
        "y": float(y),
        "roundness": float(4 * np.pi * area / perim**2) if perim > 0 else 0.0,
        "perimeter_diff": float(perim - 2 * np.sqrt(np.pi * area)),
    }


def measure_masks(masks: list[np.ndarray]) -> list[dict]:
    """Measure each object; crops to its bounding box so large images stay fast."""
    out = []
    for m in masks:
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        if not len(rows):
            continue
        t, l = max(rows[0] - 1, 0), max(cols[0] - 1, 0)
        r = measure_object(m[t:rows[-1] + 2, l:cols[-1] + 2])
        if r is not None:
            r["x"] += l
            r["y"] += t
            out.append(r)
    return out


def measure_label_mask(labels: np.ndarray) -> list[dict]:
    return measure_masks([labels == i for i in np.unique(labels) if i > 0])


def tertile_means(values: np.ndarray) -> dict:
    """Equal-count low/mid/high tiers by rank (pandas qcut on first-rank, as in the benchmark)."""
    if len(values) == 0:
        return {"low": None, "mid": None, "high": None}
    if len(values) < 3:
        v = float(np.mean(values))
        return {"low": v, "mid": v, "high": v}
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    # pandas.qcut(ranks, 3) bins at the 1/3 and 2/3 quantiles of the ranks (right-inclusive).
    q1, q2 = np.quantile(ranks, [1 / 3, 2 / 3])
    tiers = np.where(ranks <= q1, 0, np.where(ranks <= q2, 1, 2))
    return {name: float(values[tiers == k].mean()) for k, name in enumerate(("low", "mid", "high"))}


def answer(table: list[dict], metric: str, operation: str, fields: list[str]) -> dict | None:
    """Run a planned operation over the per-object table.

    operation: "max" | "min" -> the extreme object's requested fields
               "tertile_mean" -> {"low","mid","high"} means of `metric`
               "mean" | "median" | "count"
    """
    if not table:
        return None
    vals = np.array([r[metric] for r in table], dtype=float)
    if operation in ("max", "min"):
        row = table[int(np.argmax(vals) if operation == "max" else np.argmin(vals))]
        return {f: row[f] for f in fields}
    if operation == "tertile_mean":
        return tertile_means(vals)
    if operation == "mean":
        return {metric: float(vals.mean())}
    if operation == "median":
        return {metric: float(np.median(vals))}
    if operation == "count":
        return {"count": len(vals)}
    raise ValueError(f"unknown operation {operation!r}")


def tertile_means_soft(values: np.ndarray, weights: np.ndarray) -> dict:
    """Tier means where each object counts in proportion to its probability of being real.

    Hard filtering (keep only P >= 0.5) drops borderline objects, which are mostly small ones, so the
    'low' tier loses its smallest members and its mean comes out too high. Here objects are sorted
    by value and each one fills the tiers with its probability mass: tier m covers cumulative
    weight [(m-1)/3, m/3] (a weighted, Horvitz-Thompson-style version of the equal-count split).
    """
    values, weights = np.asarray(values, float), np.clip(np.asarray(weights, float), 0, None)
    if len(values) == 0 or weights.sum() <= 0:
        return {"low": None, "mid": None, "high": None}
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order] / weights.sum()
    hi = np.cumsum(w)
    lo = hi - w
    out = {}
    for m, name in enumerate(("low", "mid", "high")):
        a, b = m / 3, (m + 1) / 3
        share = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
        out[name] = float((x * share).sum() / share.sum()) if share.sum() > 0 else None
    return out


def answer_soft(table: list[dict], weights: list[float], metric: str) -> dict | None:
    """Probability-weighted tertile means (see tertile_means_soft)."""
    if not table:
        return None
    return tertile_means_soft(np.array([r[metric] for r in table]), np.array(weights))


def drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Zero out instances smaller than min_area pixels (e.g. OrganoID's own 200 px debris filter)."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    out = labels.copy()
    out[np.isin(out, ids[counts < min_area])] = 0
    return out
