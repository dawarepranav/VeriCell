"""Fetch a small, fixed sample of the LIVECell test set (images + instance annotations).

Downloads each cell type's test annotations, picks N images per type with a fixed seed,
and pulls only those images out of the 1.2 GB images.zip using HTTP range requests.
Output: data/livecell/{images/*.tif, annotations.json} (COCO format, sampled images only).

Usage:
    .venv/bin/python scripts/get_livecell.py --per-type 3
"""

import argparse
import json
import random
from pathlib import Path

import httpx
from remotezip import RemoteZip

BASE = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"
CELL_TYPES = ["a172", "bt474", "bv2", "huh7", "mcf7", "shsy5y", "skbr3", "skov3"]
OUT = Path(__file__).resolve().parent.parent / "data" / "livecell"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    images, annotations, categories = [], [], None
    for cell_type in CELL_TYPES:
        print(f"{cell_type}: annotations...", flush=True)
        coco = httpx.get(f"{BASE}/annotations/LIVECell_single_cells/{cell_type}/test.json", timeout=300).json()
        categories = categories or coco["categories"]
        # LIVECell stores "images" as a list and "annotations" as a dict keyed by id.
        anns = coco["annotations"].values() if isinstance(coco["annotations"], dict) else coco["annotations"]
        picked = rng.sample(sorted(coco["images"], key=lambda im: im["file_name"]), args.per_type)
        ids = {im["id"] for im in picked}
        for im in picked:
            images.append({**im, "cell_type": cell_type})
        annotations.extend(a for a in anns if a["image_id"] in ids)

    print(f"extracting {len(images)} images...", flush=True)
    with RemoteZip(f"{BASE}/images.zip") as z:
        for im in images:
            target = OUT / "images" / im["file_name"]
            if not target.exists():
                target.write_bytes(z.read(f"images/livecell_test_images/{im['file_name']}"))

    (OUT / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": categories}))
    print(f"done: {len(images)} images, {len(annotations)} cells -> {OUT}")


if __name__ == "__main__":
    main()
