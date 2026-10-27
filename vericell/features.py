"""Candidate pool + per-cell verification signals.

Every candidate cell comes from one source (Cellpose-SAM or SAM 3.1 text on zoomed tiles) and is
described by signals that don't need ground truth:

  agreement   does an independent model draw the same cell?          (cross_iou)
  stability   does SAM redraw it the same way from its box, zoomed?   (zoom_iou)
  confidence  SAM 3.1's own score for the cell                       (sam_conf)
  shape       size vs. the image's typical cell, solidity, roundness, elongation
  appearance  contrast with the surrounding ring, edge strength, texture
  context     crowding, overlap with same-source cells, image border
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .metrics import iou_matrix
from .sam import Instance

FEATURES = [
    "from_cellpose", "from_sam", "cross_iou", "zoom_iou", "sam_conf", "sam_conf_missing",
    "log_area_rel", "solidity", "circularity", "elongation", "contrast", "edge_strength", "texture",
    "crowding", "same_source_overlap", "touches_border", "image_density",
]


@dataclass
class Candidate:
    id: int
    source: str  # "cellpose" or "sam"
    inst: Instance
    features: dict = field(default_factory=dict)


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


def build_candidates(image: Image.Image, cellpose: list[Instance], sam_tiled: list[Instance],
                     zoom: list[Instance]) -> list[Candidate]:
    """Pool Cellpose and SAM-tiled cells and compute every verification signal for each."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    H, W = gray.shape

    cands = [Candidate(i, "cellpose", inst) for i, inst in enumerate(cellpose)]
    cands += [Candidate(len(cellpose) + i, "sam", inst) for i, inst in enumerate(sam_tiled)]
    masks = [c.inst.mask for c in cands]
    n_cp = len(cellpose)

    cross = iou_matrix(masks, [i.mask for i in cellpose] + [i.mask for i in sam_tiled])
    to_zoom = iou_matrix(masks, [z.mask for z in zoom]) if zoom else np.zeros((len(cands), 1))
    median_area = float(np.median([i.mask.sum() for i in cellpose])) if cellpose else 200.0
    centers = np.array([[(c.inst.box[0] + c.inst.box[2]) / 2, (c.inst.box[1] + c.inst.box[3]) / 2] for c in cands])
    cp_centers = centers[:n_cp]
    radius = np.sqrt(median_area / np.pi)
    density = n_cp * median_area / (H * W)

    for k, c in enumerate(cands):
        own = slice(0, n_cp) if c.source == "cellpose" else slice(n_cp, len(cands))
        other = slice(n_cp, len(cands)) if c.source == "cellpose" else slice(0, n_cp)
        same = cross[k, own].copy()
        same[k - own.start] = 0
        z = int(np.argmax(to_zoom[k])) if zoom else -1
        zoom_iou = float(to_zoom[k, z]) if zoom else 0.0
        conf = c.inst.confidence if c.source == "sam" else (zoom[z].confidence if zoom and zoom_iou > 0.3 else None)
        area = float(c.inst.mask.sum())
        solidity, circularity, elongation = _shape(c.inst.mask)
        contrast, edge_strength, texture = _appearance(gray, grad, c.inst.mask, c.inst.box)
        l, t, r, b = c.inst.box
        crowding = int((np.hypot(*(cp_centers - centers[k]).T) < 3 * radius).sum()) if n_cp else 0
        c.features = {
            "from_cellpose": float(c.source == "cellpose"),
            "from_sam": float(c.source == "sam"),
            "cross_iou": float(cross[k, other].max()) if cross[:, other].size else 0.0,
            "zoom_iou": zoom_iou,
            "sam_conf": float(conf) if conf is not None else 0.0,
            "sam_conf_missing": float(conf is None),
            "log_area_rel": float(np.log(max(area, 1.0) / median_area)),
            "solidity": solidity,
            "circularity": circularity,
            "elongation": elongation,
            "contrast": contrast,
            "edge_strength": edge_strength,
            "texture": texture,
            "crowding": float(crowding),
            "same_source_overlap": float(same.max()) if same.size else 0.0,
            "touches_border": float(l <= 0 or t <= 0 or r >= W or b >= H),
            "image_density": float(density),
        }
    return cands


def reasons(f: dict) -> list[str]:
    """Plain-language reasons a cell looks doubtful, from its signals."""
    out = []
    if f["from_sam"] and f["cross_iou"] < 0.2:
        out.append("found only by SAM (possible missed cell, or a false one)")
    elif f["from_cellpose"] and f["cross_iou"] < 0.3:
        out.append("SAM doesn't see this cell")
    elif f["cross_iou"] < 0.6:
        out.append("the two models draw it differently")
    if f["from_cellpose"] and f["zoom_iou"] < 0.6:
        out.append("mask changes when SAM redraws it zoomed in")
    if not f["sam_conf_missing"] and f["sam_conf"] < 0.5:
        out.append(f"low SAM confidence ({f['sam_conf']:.2f})")
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
