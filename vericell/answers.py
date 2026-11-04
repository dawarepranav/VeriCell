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


def stability(plan: dict, table: list, p: np.ndarray, idx: list[int], winner) -> dict | None:
    """For "most extreme" questions: could resolving an uncertain object change WHICH object is the answer?

    challengers: objects not counted yet (too unsure) that become the answer if confirmed real.
    successor:   the object that becomes the answer if the current one turns out not to be real.
    """
    if plan["operation"] not in ("max", "min") or winner is None:
        return None
    metric = plan["metric"]
    sign = 1.0 if plan["operation"] == "max" else -1.0
    wv = table[winner][metric]
    gap = lambda i: sign * (table[i][metric] - wv) / max(abs(wv), 1e-9)  # noqa: E731  >0: more extreme than the winner
    q = p.copy()
    q[winner] = 0.0
    successor = point(plan, table, q, idx)[1]
    challengers = []
    for i in idx:
        if i == winner or gap(i) <= 0:
            continue
        q = p.copy()
        q[i] = 1.0
        if point(plan, table, q, idx)[1] == i:
            challengers.append(i)
    entry = lambda i: {"id": int(i), "value": float(table[i][metric]), "p": round(float(p[i]), 3),  # noqa: E731
                       "gap": round(float(gap(i)), 4)}
    challengers.sort(key=lambda i: -p[i])
    return {"winner": entry(winner), "successor": entry(successor) if successor is not None else None,
            "challengers": [entry(i) for i in challengers[:5]], "n_challengers": len(challengers)}


def headline_key(plan: dict) -> str | None:
    """The answer field a single-number claim is about (None for three-tier answers)."""
    if plan["operation"] == "tertile_mean":
        return None
    return "count" if plan["operation"] == "count" else plan["metric"]


def min_flip(plan: dict, table: list, p: np.ndarray, idx: list[int], candidates: list[int],
             op: str, threshold: float, max_k: int = 8) -> dict:
    """Try to break a claim ("answer at most / at least threshold") by resolving uncertain objects.

    Greedy: each step resolves one object as real or not real, taking a move that breaks the claim if
    one exists (the most likely such move), otherwise the move that pushes the answer furthest toward
    breaking it. Returns the smallest breaking set found, which is not guaranteed to be the minimum.
    """
    key = headline_key(plan)
    holds = (lambda v: v <= threshold) if op == "at_most" else (lambda v: v >= threshold)  # noqa: E731
    toward = 1.0 if op == "at_most" else -1.0  # direction that breaks the claim
    value = lambda q: (point(plan, table, q, idx)[0] or {}).get(key)  # noqa: E731
    q = p.copy()
    v0 = value(q)
    out = {"value_now": v0, "holds_now": v0 is not None and holds(v0), "flips": [], "broken": False,
           "searched": len(candidates)}
    if not out["holds_now"]:
        return out
    left, current = list(candidates), v0
    for _ in range(max_k):
        best = None  # (breaks, score, likelihood, i, state, value)
        for i in left:
            for state in (0.0, 1.0):
                r = q.copy()
                r[i] = state
                v = value(r)
                if v is None:
                    continue
                like = float(p[i] if state else 1.0 - p[i])
                cand = (not holds(v), toward * (v - current), like, i, state, v)
                if best is None or cand[:3] > best[:3]:
                    best = cand
        if best is None or (not best[0] and best[1] <= 0):
            break  # nothing moves the answer toward breaking the claim
        _, _, like, i, state, v = best
        q[i] = state
        left.remove(i)
        current = v
        out["flips"].append({"id": int(i), "to_real": bool(state), "p": round(float(p[i]), 3),
                             "likelihood": round(like, 3), "value_after": float(v)})
        if not holds(v):
            out["broken"] = True
            break
    out["value_after"] = current
    out["likelihood"] = float(np.prod([f["likelihood"] for f in out["flips"]])) if out["broken"] else None
    return out


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
