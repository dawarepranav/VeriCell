"""Instance proposers: Cellpose-SAM (single scale or a resolution pyramid), and Cellpose boxes refined by SAM 3.1."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from .sam import SAM, Instance


@lru_cache(maxsize=1)
def _cellpose_model():
    import torch
    from cellpose import models

    return models.CellposeModel(gpu=torch.backends.mps.is_available() or torch.cuda.is_available())


def labels_to_instances(labels: np.ndarray, prompt: str) -> list[Instance]:
    out = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        mask = labels == lab
        ys, xs = np.nonzero(mask)
        out.append(Instance(mask=mask, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                            confidence=None, prompt=prompt))
    return out


def cellpose(image: Image.Image) -> list[Instance]:
    labels, _flows, _styles = _cellpose_model().eval(np.asarray(image))
    return labels_to_instances(labels, "cellpose")


def hybrid(image: Image.Image, sam: SAM, base: list[Instance] | None = None, pad: int = 2) -> list[Instance]:
    """Cellpose finds the cells, SAM 3.1 re-segments each one from its (padded) box."""
    w, h = image.size
    base = cellpose(image) if base is None else base
    boxes = [(max(0, l - pad), max(0, t - pad), min(w, r + pad), min(h, b + pad))
             for l, t, r, b in (i.box for i in base)]
    return sam.segment_boxes(image, boxes)


def _upsample_instance(coarse: np.ndarray, scale: float, shape: tuple[int, int]) -> np.ndarray | None:
    """Upsample one coarse binary mask to full resolution through its signed distance field,
    so edges come out smooth instead of blocky (nearest-neighbour would add staircase steps)."""
    import cv2
    from scipy.ndimage import distance_transform_edt

    rows, cols = np.flatnonzero(coarse.any(1)), np.flatnonzero(coarse.any(0))
    if not len(rows):
        return None
    t, b = max(rows[0] - 2, 0), min(rows[-1] + 3, coarse.shape[0])
    l, r = max(cols[0] - 2, 0), min(cols[-1] + 3, coarse.shape[1])
    crop = coarse[t:b, l:r]
    sdf = distance_transform_edt(crop) - distance_transform_edt(~crop)
    H, W = shape
    ft, fb = int(round(t / scale)), min(H, int(round(b / scale)))
    fl, fr = int(round(l / scale)), min(W, int(round(r / scale)))
    if fb <= ft or fr <= fl:
        return None
    fine = cv2.resize(sdf.astype(np.float32), (fr - fl, fb - ft), interpolation=cv2.INTER_LINEAR) > 0
    full = np.zeros(shape, dtype=bool)
    full[ft:fb, fl:fr] = fine
    return full if full.any() else None


def cellpose_scaled(image: Image.Image, scale: float) -> list[Instance]:
    """Cellpose-SAM on a downscaled copy of the image, masks mapped back to full resolution.

    Cellpose-SAM was trained on objects ~8-120 px across (centred on 30 px) in 256 px windows, and its
    flow integration (200 steps) can't carry pixels across a 300 px organoid, so big organoids break
    into shards. Downscaling brings them back into range. Coarse passes use a looser flow-error
    threshold: real organoids with lumens produce flows that look 'bad' to the default 0.4 check.
    """
    import cv2

    arr = np.asarray(image)
    if scale == 1.0:
        labels, _f, _s = _cellpose_model().eval(arr)
        return labels_to_instances(labels, "cellpose@1")
    small = cv2.resize(arr, (max(1, round(arr.shape[1] * scale)), max(1, round(arr.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    labels, _f, _s = _cellpose_model().eval(small, flow_threshold=0.6, cellprob_threshold=-0.5)
    out = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        full = _upsample_instance(labels == lab, scale, arr.shape[:2])
        if full is not None:
            ys, xs = np.nonzero(full)
            out.append(Instance(mask=full, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                                confidence=None, prompt=f"cellpose@{scale:.3g}"))
    return out


def pyramid_scales(image: Image.Image, min_side: int = 160) -> list[float]:
    """Native resolution plus 1/3 and 1/8 copies, skipping any copy smaller than min_side pixels."""
    side = min(image.size)
    return [s for s in (1.0, 1 / 3, 1 / 8) if s == 1.0 or side * s >= min_side]


def _iomin(a: Instance, b: Instance) -> float:
    l, t = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    r, btm = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    if l >= r or t >= btm:
        return 0.0
    inter = np.logical_and(a.mask[t:btm, l:r], b.mask[t:btm, l:r]).sum()
    return float(inter / min(a.mask.sum(), b.mask.sum()))


def merge_scales(by_scale: dict[float, list[Instance]], contain: float = 0.65, tiled: float = 0.6) -> list[Instance]:
    """Combine pyramid levels, coarse to fine, using containment (IoMin) instead of IoU.

    For each coarse object, look at the finer objects it contains:
      - none                          -> keep the coarse object (the finer pass missed it)
      - one near-duplicate            -> keep the finer one (sharper edges at native resolution)
      - several covering most of it   -> it's a merge of real objects: keep the finer ones
      - several covering little of it -> they're shards of one big object: keep the coarse one
    """
    scales = sorted(by_scale)  # coarse (small scale) first
    kept = list(by_scale[scales[0]])
    for s in scales[1:]:
        fine = by_scale[s]
        drop_fine: set[int] = set()
        drop_kept: set[int] = set()
        for k, c in enumerate(kept):
            inside = [i for i, f in enumerate(fine) if _iomin(f, c) > contain]
            if not inside:
                continue
            area_c = c.mask.sum()
            cover = np.logical_or.reduce([fine[i].mask for i in inside]).sum() / area_c
            if len(inside) == 1:
                f = fine[inside[0]]
                iou = np.logical_and(f.mask, c.mask).sum() / np.logical_or(f.mask, c.mask).sum()
                (drop_kept if iou > 0.7 else drop_fine).add(k if iou > 0.7 else inside[0])
            elif cover >= tiled:
                drop_kept.add(k)
            else:
                drop_fine.update(inside)
        kept = [c for k, c in enumerate(kept) if k not in drop_kept] + \
               [f for i, f in enumerate(fine) if i not in drop_fine]
    return kept


def cellpose_multiscale(image: Image.Image) -> tuple[list[Instance], dict[float, list[Instance]]]:
    by_scale = {s: cellpose_scaled(image, s) for s in pyramid_scales(image)}
    return merge_scales(by_scale), by_scale


def touching_clusters(instances: list[Instance], gap: int = 2, max_parts: int = 8) -> list[Instance]:
    """Join instances that touch (within `gap` px) into single candidates.

    A budding organoid often comes out of Cellpose as one object per lobe; the joined version is
    offered as an extra candidate and the verifier decides which reading is right.
    """
    import cv2
    from scipy.sparse.csgraph import connected_components

    n = len(instances)
    if n < 2:
        return []
    kernel = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    adj = np.zeros((n, n), dtype=bool)
    for i, a in enumerate(instances):
        l, t, r, b = a.box
        for j in range(i + 1, n):
            c = instances[j]
            if c.box[0] > r + gap or c.box[2] < l - gap or c.box[1] > b + gap or c.box[3] < t - gap:
                continue
            L, T = max(0, min(l, c.box[0]) - gap), max(0, min(t, c.box[1]) - gap)
            R, B = max(r, c.box[2]) + gap, max(b, c.box[3]) + gap
            grown = cv2.dilate(a.mask[T:B, L:R].astype(np.uint8), kernel) > 0
            adj[i, j] = adj[j, i] = bool((grown & c.mask[T:B, L:R]).any())
    k, labels = connected_components(adj, directed=False)
    out = []
    for comp in range(k):
        members = np.flatnonzero(labels == comp)
        if not 2 <= len(members) <= max_parts:
            continue
        mask = np.logical_or.reduce([instances[m].mask for m in members])
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0  # close the seams
        ys, xs = np.nonzero(mask)
        out.append(Instance(mask=mask, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                            confidence=None, prompt=f"merged:{len(members)}"))
    return out
