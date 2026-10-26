"""Benchmark loaders. Each sample is (name, PIL RGB image, list of ground-truth bool masks, meta)."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

DATA = Path(__file__).resolve().parent.parent / "data"


def livecell(root: Path = DATA / "livecell"):
    coco = json.loads((root / "annotations.json").read_text())
    by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        h, w = im["height"], im["width"]
        gt = []
        for a in by_image.get(im["id"], []):
            rle = mask_utils.frPyObjects(a["segmentation"], h, w)
            gt.append(mask_utils.decode(mask_utils.merge(rle) if isinstance(rle, list) else rle).astype(bool))
        image = Image.open(root / "images" / im["file_name"]).convert("RGB")
        yield im["file_name"], image, gt, {"cell_type": im["cell_type"]}
