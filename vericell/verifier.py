"""Per-cell verifier, fusion, review queue and conformal count intervals."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .features import FEATURES, Candidate


REVIEW_BAND = (0.3, 0.7)  # cells whose P(correct) falls here go to the human review queue


def matrix(cands: list[Candidate]) -> np.ndarray:
    return np.array([[c.features[f] for f in FEATURES] for c in cands], dtype=float)


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Gradient boosting, isotonic-calibrated on held-out images inside the training set."""
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)
    splits = list(GroupKFold(n_splits=min(3, len(np.unique(groups)))).split(X, y, groups))
    return CalibratedClassifierCV(base, method="isotonic", cv=splits).fit(X, y)


def out_of_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 6) -> np.ndarray:
    """P(correct) for every cell from a model that never saw that cell's image."""
    p = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=n_folds).split(X, y, groups):
        p[test] = fit(X[train], y[train], groups[train]).predict_proba(X[test])[:, 1]
    return p


def fuse(cands: list[Candidate], p: np.ndarray, threshold: float = 0.5, max_overlap: float = 0.3) -> np.ndarray:
    """Greedy selection, most trusted first: keep a cell if P >= threshold and it doesn't overlap a kept cell.

    Returns the indices of accepted candidates.
    """
    order = np.argsort(-p)
    kept: list[int] = []
    boxes = [c.inst.box for c in cands]
    areas = [int(c.inst.mask.sum()) for c in cands]
    for i in order:
        if p[i] < threshold:
            break
        clash = False
        for k in kept:
            (l1, t1, r1, b1), (l2, t2, r2, b2) = boxes[i], boxes[k]
            l, t, r, b = max(l1, l2), max(t1, t2), min(r1, r2), min(b1, b2)
            if l >= r or t >= b:
                continue
            inter = int(np.logical_and(cands[i].inst.mask[t:b, l:r], cands[k].inst.mask[t:b, l:r]).sum())
            if inter / (areas[i] + areas[k] - inter) > max_overlap or inter / min(areas[i], areas[k]) > 0.5:
                clash = True
                break
        if not clash:
            kept.append(int(i))
    return np.array(kept, dtype=int)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error and the reliability-diagram bins."""
    edges = np.linspace(0, 1, bins + 1)
    rows, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if sel.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                         "mean_p": float(p[sel].mean()), "frac_correct": float(y[sel].mean())})
            total += sel.sum() / len(p) * abs(p[sel].mean() - y[sel].mean())
    return float(total), rows


def conformal_counts(pred: np.ndarray, true: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-image-out split-conformal intervals for per-image cell counts.

    Score = |log(true / pred)|; each image's interval uses the other images' scores, so the
    guarantee is (1 - alpha) coverage for a new image from the same distribution.
    """
    scores = np.abs(np.log(true / pred))
    lo, hi = np.zeros(len(pred)), np.zeros(len(pred))
    for i in range(len(pred)):
        rest = np.delete(scores, i)
        k = min(len(rest) - 1, int(np.ceil((len(rest) + 1) * (1 - alpha))) - 1)
        q = np.sort(rest)[k]
        lo[i], hi[i] = pred[i] * np.exp(-q), pred[i] * np.exp(q)
    return lo, hi
