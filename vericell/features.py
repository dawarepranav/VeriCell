"""Candidate pool + per-cell verification signals.

Every candidate cell comes from one source (Cellpose-SAM or SAM 3.1 text on zoomed tiles) and is
described by signals that don't need ground truth:

  agreement   does an independent model draw the same cell?          (cross_iou, plus size-aware
              versions: IoMin, and IoU relative to what a 1.5 px edge offset would give at this size)
  stability   does SAM redraw it the same way from its box, zoomed?   (zoom_iou)
  confidence  SAM 3.1's own score for the cell                       (sam_conf)
  shape       size vs. the image's typical cell, solidity, roundness, elongation
  appearance  contrast with the surrounding ring, edge strength, texture
  context     crowding, overlap with same-source cells, image border
  judge       Muse Spark's verdict on a crop, where one was asked (confidence-weighted; see judge.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .metrics import iou_matrix
from .sam import Instance

FEATURES = [
    "from_cellpose", "from_sam", "from_coarse", "from_merged", "cross_iou", "cross_iomin", "cross_iou_norm",
    "scale_agree", "zoom_iou", "sam_conf", "sam_conf_missing",
    "log_area_rel", "solidity", "circularity", "elongation", "contrast", "edge_strength", "texture",
    "crowding", "same_source_overlap", "touches_border", "image_density",
    "judge_one", "judge_several", "judge_not", "judge_missing",
]


@dataclass
class Candidate:
    id: int
    source: str  # "cellpose", "coarse" (Cellpose at a coarser scale), "merged" (touching parts joined), "sam"
    inst: Instance
    features: dict = field(default_factory=dict)
    verdict: dict | None = None


def _shape(mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area = float(mask.sum())
    hull = cv2.contourArea(cv2.convexHull(c)) or area
    perim = cv2.arcLength(c, True) or 1.0
    (_, _), (w, h), _ = cv2.minAreaRect(c) if len(c) >= 3 else ((0, 0), (1, 1), 0)
    return (min(1.0, area / hull), min(1.5, 4 * np.pi * area / perim**2),
            float(max(w, h) / max(min(w, h), 1.0)))


def _appearance(gray: np.ndarray, grad: np.ndarray, mask: np.ndarray, box) -> tuple[float, float, float]:
    h, w = gray.shape
    pad = 6
    l, t, r, b = max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)
    m = mask[t:b, l:r].astype(np.uint8)
    g = gray[t:b, l:r]
    ring = (cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0) & (m == 0)
    edge = (cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0)
    inside = g[m > 0]
    contrast = float(abs(inside.mean() - g[ring].mean())) if ring.any() else 0.0
    return contrast, float(grad[t:b, l:r][edge].mean()) if edge.any() else 0.0, float(inside.std())


def _overlaps(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """Pairwise intersection counts, computed only inside overlapping bounding boxes."""
    def boxes(ms):
        out = []
        for m in ms:
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            out.append((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1) if len(rows) else (0, 0, 0, 0))
        return np.array(out).reshape(-1, 4)

    inter = np.zeros((len(a), len(b)))
    if not a or not b:
        return inter
    ba, bb = boxes(a), boxes(b)
    for i, (m, (l, t, r, btm)) in enumerate(zip(a, ba)):
        hits = np.nonzero((bb[:, 0] < r) & (bb[:, 2] > l) & (bb[:, 1] < btm) & (bb[:, 3] > t))[0]
        for j in hits:
            L, T, R, B = max(l, bb[j, 0]), max(t, bb[j, 1]), min(r, bb[j, 2]), min(btm, bb[j, 3])
            inter[i, j] = np.logical_and(m[T:B, L:R], b[j][T:B, L:R]).sum()
    return inter


def build_candidates(image: Image.Image, cellpose: list[Instance], sam_tiled: list[Instance],
                     zoom: list[Instance], extra: dict[str, list[Instance]] | None = None) -> list[Candidate]:
    """Pool every proposer's cells and compute every verification signal for each.

    `extra` adds more Cellpose-family candidates: "coarse" (Cellpose on downscaled copies) and
    "merged" (touching Cellpose parts joined into one object, e.g. a budding organoid's lobes).
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    H, W = gray.shape

    sources = [("cellpose", cellpose)] + list((extra or {}).items()) + [("sam", sam_tiled)]
    cands: list[Candidate] = []
    for name, insts in sources:
        cands += [Candidate(len(cands) + i, name, inst) for i, inst in enumerate(insts)]
    masks = [c.inst.mask for c in cands]
    src = np.array([c.source for c in cands])
    sam_family = src == "sam"
    areas = np.array([m.sum() for m in masks], dtype=float)

    inter = _overlaps(masks, masks)
    union = areas[:, None] + areas[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    iomin = np.divide(inter, np.minimum(areas[:, None], areas[None, :]),
                      out=np.zeros_like(inter), where=inter > 0)
    to_zoom = iou_matrix(masks, [z.mask for z in zoom]) if zoom else np.zeros((len(cands), 1))
    median_area = float(np.median([i.mask.sum() for i in cellpose])) if cellpose else 200.0
    centers = np.array([[(c.inst.box[0] + c.inst.box[2]) / 2, (c.inst.box[1] + c.inst.box[3]) / 2] for c in cands])
    cp_centers = centers[src == "cellpose"]
    radius = np.sqrt(median_area / np.pi)
    density = len(cellpose) * median_area / (H * W)

    for k, c in enumerate(cands):
        other = sam_family != sam_family[k]                    # the independent model family
        same = (src == src[k]) & (np.arange(len(cands)) != k)  # same proposer, other cells
        sibling = (~sam_family) & (src != src[k]) if not sam_family[k] else np.zeros(len(cands), bool)
        r_eq = np.sqrt(areas[k] / np.pi)
        cross_iou = float(iou[k, other].max()) if other.any() else 0.0
        z = int(np.argmax(to_zoom[k])) if zoom else -1
        zoom_iou = float(to_zoom[k, z]) if zoom else 0.0
        conf = c.inst.confidence if c.source == "sam" else (zoom[z].confidence if zoom and zoom_iou > 0.3 else None)
        solidity, circularity, elongation = _shape(c.inst.mask)
        contrast, edge_strength, texture = _appearance(gray, grad, c.inst.mask, c.inst.box)
        l, t, r, b = c.inst.box
        crowding = int((np.hypot(*(cp_centers - centers[k]).T) < 3 * radius).sum()) if len(cp_centers) else 0
        c.features = {
            "from_cellpose": float(c.source == "cellpose"),
            "from_sam": float(c.source == "sam"),
            "from_coarse": float(c.source == "coarse"),
            "from_merged": float(c.source == "merged"),
            "cross_iou": cross_iou,
            "cross_iomin": float(iomin[k, other].max()) if other.any() else 0.0,
            # IoU a perfect match would still lose to a 1.5 px edge offset at this size: (R / (R + 1.5))^2
            "cross_iou_norm": float(min(1.5, cross_iou / (r_eq / (r_eq + 1.5)) ** 2)) if r_eq > 0 else 0.0,
            "scale_agree": float(iou[k, sibling].max()) if sibling.any() else 0.0,
            "zoom_iou": zoom_iou,
            "sam_conf": float(conf) if conf is not None else 0.0,
            "sam_conf_missing": float(conf is None),
            "log_area_rel": float(np.log(max(areas[k], 1.0) / median_area)),
            "solidity": solidity,
            "circularity": circularity,
            "elongation": elongation,
            "contrast": contrast,
            "edge_strength": edge_strength,
            "texture": texture,
            "crowding": float(crowding),
            "same_source_overlap": float(iou[k, same].max()) if same.any() else 0.0,
            "touches_border": float(l <= 0 or t <= 0 or r >= W or b >= H),
            "image_density": float(density),
            "judge_one": 0.0, "judge_several": 0.0, "judge_not": 0.0, "judge_missing": 1.0,
        }
    return cands


def add_judgement(c: Candidate, verdict: dict) -> None:
    """Record a crop-judge verdict on a candidate as verifier signals."""
    conf = float(verdict["confidence"])
    c.features.update({"judge_one": conf * (verdict["verdict"] == "one_object"),
                       "judge_several": conf * (verdict["verdict"] == "several_objects"),
                       "judge_not": conf * (verdict["verdict"] == "not_an_object"),
                       "judge_missing": 0.0})
    c.verdict = verdict


def reasons(f: dict) -> list[str]:
    """Plain-language reasons a cell looks doubtful, from its signals."""
    out = []
    if f["from_sam"] and f["cross_iou"] < 0.2:
        out.append("found only by SAM (possible missed cell, or a false one)")
    elif not f["from_sam"] and f["cross_iou"] < 0.3:
        out.append("SAM doesn't see this cell")
    elif f["cross_iou"] < 0.6:
        out.append("the two models draw it differently")
    if f["from_cellpose"] and f["zoom_iou"] < 0.6:
        out.append("mask changes when SAM redraws it zoomed in")
    if not f["sam_conf_missing"] and f["sam_conf"] < 0.5:
        out.append(f"low SAM confidence ({f['sam_conf']:.2f})")
    if f["from_merged"]:
        if f["judge_several"] > 0.5:
            out.append("Muse Spark thinks these are several separate objects")
        elif f["judge_one"] > 0.5:
            out.append("touching parts joined; Muse Spark confirms it's one object")
        else:
            out.append("several touching parts joined into one object (one budding organoid, or several?)")
    if f["log_area_rel"] > 1.0:
        out.append("much larger than a typical cell (possible merge)")
    elif f["log_area_rel"] < -1.2:
        out.append("much smaller than a typical cell (possible fragment)")
    if f["solidity"] < 0.85:
        out.append("irregular, concave outline (possible merge)")
    if f["same_source_overlap"] > 0.2:
        out.append("overlaps another cell")
    if f["touches_border"]:
        out.append("cut off by the image edge")
    return out
