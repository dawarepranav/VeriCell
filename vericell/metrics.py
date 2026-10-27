"""Instance matching and segmentation metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(pred: list[np.ndarray], gt: list[np.ndarray]) -> np.ndarray:
    """Pairwise IoU, computed only for pairs whose bounding boxes overlap."""
    def boxes(ms):
        out = []
        for m in ms:
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            out.append((cols[0], rows[0], cols[-1], rows[-1]) if len(rows) else (0, 0, -1, -1))
        return np.array(out).reshape(-1, 4)

    iou = np.zeros((len(pred), len(gt)))
    if not pred or not gt:
        return iou
    pb, gb = boxes(pred), boxes(gt)
    areas_p = [m.sum() for m in pred]
    areas_g = [g.sum() for g in gt]
    for i, (m, b) in enumerate(zip(pred, pb)):
        hits = np.nonzero((gb[:, 0] <= b[2]) & (gb[:, 2] >= b[0]) & (gb[:, 1] <= b[3]) & (gb[:, 3] >= b[1]))[0]
        for j in hits:
            # Intersection can only lie inside the overlap of the two boxes.
            l, t = max(b[0], gb[j, 0]), max(b[1], gb[j, 1])
            r, btm = min(b[2], gb[j, 2]) + 1, min(b[3], gb[j, 3]) + 1
            inter = np.logical_and(m[t:btm, l:r], gt[j][t:btm, l:r]).sum()
            if inter:
                iou[i, j] = inter / (areas_p[i] + areas_g[j] - inter)
    return iou


def match(iou: np.ndarray, thresh: float = 0.5):
    """One-to-one matching. Returns (pred_best_iou per prediction, list of (pred, gt) pairs >= thresh)."""
    if iou.size == 0:
        return np.zeros(iou.shape[0]), []
    rows, cols = linear_sum_assignment(-iou)
    pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= thresh]
    best = np.zeros(iou.shape[0])
    for r, c in zip(rows, cols):
        best[r] = iou[r, c]
    return best, pairs


def scores(n_pred: int, n_gt: int, pairs, iou: np.ndarray) -> dict:
    tp = len(pairs)
    return {
        "n_pred": n_pred, "n_gt": n_gt, "tp": tp,
        "precision": tp / n_pred if n_pred else 0.0,
        "recall": tp / n_gt if n_gt else 0.0,
        "f1": 2 * tp / (n_pred + n_gt) if n_pred + n_gt else 0.0,
        "mean_matched_iou": float(np.mean([iou[r, c] for r, c in pairs])) if pairs else 0.0,
    }


def auroc(scores_: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores_` for predicting `labels` (1 = correct). Ties count half."""
    pos, neg = scores_[labels == 1], scores_[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
