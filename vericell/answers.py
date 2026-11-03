"""Answers with calibrated 90% ranges.

Point answer: deterministic geometry over the verified objects (strict set for "most extreme"
questions, probability-weighted pool for tiers/counts/means).

Range: split conformal, learned per question type on held-out benchmark images, in one of two forms
(calibration keeps whichever is narrower at the same 90% coverage):
  "mc"   point +- q * (Monte Carlo half-width + 2% of |point|). The Monte Carlo spread over each
         object's P(real) captures doubt about *which detected objects exist*, so this range
         narrows as a human reviews objects; q widens it to cover what it can't see (missed
         objects, boundary error).
  "rel"  point +- q * |point|. Better when the error is mostly systematic (e.g. counts, where
         missed and merged objects dominate).
"""

from __future__ import annotations

import numpy as np

from .measure import answer, tertile_means_soft
from .verifier import fuse

STRICT_EXTREME_P = 0.7
POOL_P = 0.05
N_SAMPLES = 300
FLOOR = 0.02  # every range is at least +-2% x q of the answer: no answer is ever "exact"


def pool(cands, p: np.ndarray, table: list) -> list[int]:
    return [int(i) for i in fuse(cands, p, threshold=POOL_P) if table[i] is not None]


def point(plan: dict, table: list, p: np.ndarray, idx: list[int]):
    """(answer dict, index of the object the answer is about or None)."""
    metric, op = plan["metric"], plan["operation"]
    if not idx:
        return None, None
    if op in ("max", "min"):
        strong = [i for i in idx if p[i] >= STRICT_EXTREME_P] or [i for i in idx if p[i] >= 0.5] or idx
        vals = [table[i][metric] for i in strong]
        k = int(np.argmax(vals) if op == "max" else np.argmin(vals))
        return answer([table[strong[k]]], metric, op, plan["fields"]), strong[k]
    w = p[idx]
    if op == "count":
        return {"count": float(w.sum())}, None
    vals = np.array([table[i][metric] for i in idx])
    if op == "tertile_mean":
        return tertile_means_soft(vals, w), None
    if op == "mean":
        return {metric: float((vals * w).sum() / w.sum())}, None
    order = np.argsort(vals)
    cdf = np.cumsum(w[order]) / w.sum()
    return {metric: float(vals[order][min(len(vals) - 1, np.searchsorted(cdf, 0.5))])}, None


def mc_halfwidth(plan: dict, table: list, p: np.ndarray, idx: list[int], seed: int = 0) -> dict:
    """Half-width of the 5-95% Monte Carlo band per answer field (existence uncertainty only)."""
    metric, op = plan["metric"], plan["operation"]
    rng = np.random.default_rng(seed)
    vals = np.array([table[i][metric] for i in idx]) if metric != "count" and idx else None
    samples = []
    for _ in range(N_SAMPLES):
        alive = rng.random(len(idx)) < p[idx]
        if not alive.any():
            continue
        if op in ("max", "min"):
            samples.append(answer([table[idx[k]] for k in np.flatnonzero(alive)], metric, op, plan["fields"]))
        elif op == "tertile_mean":
            samples.append(tertile_means_soft(vals[alive], np.ones(alive.sum())))
        elif op == "count":
            samples.append({"count": float(alive.sum())})
        elif op == "mean":
            samples.append({metric: float(vals[alive].mean())})
        else:
            samples.append({metric: float(np.median(vals[alive]))})
    out = {}
    for s in samples[:1]:
        for k in s:
            xs = [x[k] for x in samples if x.get(k) is not None]
            if xs:
                lo, hi = np.percentile(xs, (5, 95))
                out[k] = float((hi - lo) / 2)
    return out


def calibration_key(plan: dict, field: str) -> str:
    return f"{plan['operation']}:{plan['metric']}:{'xy' if field in ('x', 'y') else 'value'}"


def halfwidth(kind: str, v: float, hw: float) -> float:
    return abs(v) if kind == "rel" else hw + FLOOR * abs(v)


def ranges(plan: dict, value: dict, hw: dict, calib: dict | None) -> dict:
    """Calibrated 90% range per field (see module docstring)."""
    out = {}
    for k, v in (value or {}).items():
        if v is None:
            continue
        c = (calib or {}).get(calibration_key(plan, k), {"q": 1.645, "kind": "mc"})  # uncalibrated: ~Gaussian 90%
        half = c["q"] * halfwidth(c["kind"], v, hw.get(k, 0.0))
        lo, hi = v - half, v + half
        if plan["operation"] == "count" or plan["metric"] in ("Area_outer", "Perimeter_outer"):
            lo = max(lo, 0.0)
        out[k] = [float(lo), float(hi)]
    return out
