"""Instance proposers: Cellpose-SAM alone, and Cellpose boxes refined by SAM 3.1 (the hybrid)."""

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
