"""Baseline numbers on the LIVECell sample: SAM 3.1 text prompt, Cellpose-SAM, and the hybrid.

Predictions are cached in outputs/cache/ (COCO RLE) so re-running doesn't call the API again.
Per-instance records go to outputs/baseline_instances.jsonl for the verifier work.

Usage:
    .venv/bin/python scripts/baseline.py [--concept cell] [--limit N]
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vericell import api, data, proposers  # noqa: E402
from vericell.metrics import auroc, iou_matrix, match, scores  # noqa: E402
from vericell.sam import SAM, Instance  # noqa: E402

CACHE = ROOT / "outputs" / "cache"


def save(path: Path, instances: list[Instance]) -> None:
    rows = []
    for i in instances:
        rle = mask_utils.encode(np.asfortranarray(i.mask.astype(np.uint8)))
        rows.append({"rle": {"size": rle["size"], "counts": rle["counts"].decode()},
                     "box": i.box, "confidence": i.confidence, "prompt": i.prompt})
    path.write_text(json.dumps(rows))


def load(path: Path) -> list[Instance]:
    out = []
    for r in json.loads(path.read_text()):
        rle = {"size": r["rle"]["size"], "counts": r["rle"]["counts"].encode()}
        out.append(Instance(mask=mask_utils.decode(rle).astype(bool), box=tuple(r["box"]),
                            confidence=r["confidence"], prompt=r["prompt"]))
    return out


def cached(name: str, method: str, fn) -> list[Instance]:
    path = CACHE / f"{Path(name).stem}.{method}.json"
    if path.exists():
        return load(path)
    instances = fn()
    save(path, instances)
    return instances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="cell")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    sam = SAM(api.client())
    samples = list(data.livecell())[: args.limit]

    # Cellpose runs locally, one image at a time; SAM calls go out in parallel.
    preds = {}
    for name, image, _gt, _meta in samples:
        preds[(name, "cellpose")] = cached(name, "cellpose", lambda: proposers.cellpose(image))
        print(f"cellpose {name}: {len(preds[(name, 'cellpose')])}", flush=True)

    text_method = f"sam_text_{args.concept}"

    def run_sam(sample):
        name, image, _gt, _meta = sample
        t = cached(name, text_method, lambda: sam.segment_text(image, args.concept))
        h = cached(name, "hybrid", lambda: proposers.hybrid(image, sam, base=preds[(name, "cellpose")]))
        print(f"sam {name}: text {len(t)}, hybrid {len(h)}", flush=True)
        return name, t, h

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, t, h in pool.map(run_sam, samples):
            preds[(name, text_method)] = t
            preds[(name, "hybrid")] = h

    methods = [text_method, "cellpose", "hybrid"]
    totals = {m: defaultdict(float) for m in methods}
    by_type = {m: defaultdict(lambda: defaultdict(float)) for m in methods}
    records = []
    for name, _image, gt, meta in samples:
        for m in methods:
            inst = preds[(name, m)]
            iou = iou_matrix([i.mask for i in inst], gt)
            best, pairs = match(iou)
            s = scores(len(inst), len(gt), pairs, iou)
            for k in ("n_pred", "n_gt", "tp"):
                totals[m][k] += s[k]
                by_type[m][meta["cell_type"]][k] += s[k]
            totals[m]["iou_sum"] += s["mean_matched_iou"] * s["tp"]
            for i, b in zip(inst, best):
                records.append({"image": name, "cell_type": meta["cell_type"], "method": m,
                                "confidence": i.confidence, "best_iou": float(b),
                                "area": int(i.mask.sum()), "correct": bool(b >= 0.5)})

    out = ROOT / "outputs"
    with open(out / "baseline_instances.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)

    def f1(t):
        return 2 * t["tp"] / (t["n_pred"] + t["n_gt"]) if t["n_pred"] + t["n_gt"] else 0.0

    types = sorted({meta["cell_type"] for *_, meta in samples})
    print(f"\nLIVECell sample: {len(samples)} images, {int(totals[methods[0]]['n_gt'])} GT cells (F1 @ IoU 0.5)\n")
    print(f"{'method':22} {'pred':>6} {'TP':>6} {'prec':>6} {'recall':>6} {'F1':>6} {'mIoU':>6}  " +
          " ".join(f"{t:>6}" for t in types))
    for m in methods:
        t = totals[m]
        print(f"{m:22} {int(t['n_pred']):6d} {int(t['tp']):6d} {t['tp'] / max(t['n_pred'], 1):6.3f} "
              f"{t['tp'] / t['n_gt']:6.3f} {f1(t):6.3f} {t['iou_sum'] / max(t['tp'], 1):6.3f}  " +
              " ".join(f"{f1(by_type[m][ct]):6.3f}" for ct in types))

    print("\nCan SAM's own confidence tell right cells from wrong ones? (AUROC, 0.5 = chance)")
    for m in (text_method, "hybrid"):
        rs = [r for r in records if r["method"] == m and r["confidence"] is not None]
        if rs:
            c = np.array([r["confidence"] for r in rs])
            y = np.array([r["correct"] for r in rs], dtype=int)
            print(f"  {m:22} AUROC {auroc(c, y):.3f}  (n={len(rs)}, {y.mean():.1%} correct)")


if __name__ == "__main__":
    main()
