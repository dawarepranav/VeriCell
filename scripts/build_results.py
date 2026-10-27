"""Train/evaluate the per-cell verifier on the LIVECell sample and export everything the UI shows.

Every number is out-of-fold: each image is scored by a verifier trained on other images only.

Usage:
    .venv/bin/python scripts/run_candidates.py   # once; cached
    .venv/bin/python scripts/build_results.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from vericell import data  # noqa: E402
from vericell.features import FEATURES, build_candidates, reasons  # noqa: E402
from vericell.metrics import auroc, iou_matrix, match  # noqa: E402
from vericell.verifier import REVIEW_BAND, conformal_counts, ece, fuse, matrix, out_of_fold  # noqa: E402

CACHE = ROOT / "outputs" / "cache"
UI = ROOT / "outputs" / "ui"
SAM_SIGNALS = {"cross_iou", "zoom_iou", "sam_conf", "sam_conf_missing", "from_sam"}


def polygon(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.nonzero(mask)
    l, t = xs.min(), ys.min()
    crop = mask[t:ys.max() + 1, l:xs.max() + 1].astype(np.uint8)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.6, True)[:, 0, :]
    return [[int(x + l), int(y + t)] for x, y in c]


def prf(n_pred: int, n_gt: int, tp: int) -> dict:
    return {"precision": tp / max(n_pred, 1), "recall": tp / max(n_gt, 1), "f1": 2 * tp / max(n_pred + n_gt, 1),
            "n_pred": n_pred, "tp": tp}


def score(masks: list[np.ndarray], gt: list[np.ndarray]) -> int:
    return len(match(iou_matrix(masks, gt))[1])


def main() -> None:
    images = []
    for name, image, gt, meta in data.livecell():
        stem = name[:-4]
        cp, tx, zm = (load(CACHE / f"{stem}.{m}.json") for m in ("cellpose", "sam_text_tiled", "hybrid_zoom"))
        cands = build_candidates(image, cp, tx, zm)
        best = iou_matrix([c.inst.mask for c in cands], gt).max(1) if gt else np.zeros(len(cands))
        images.append({"name": name, "stem": stem, "image": image, "gt": gt, "meta": meta,
                       "cp": cp, "tx": tx, "cands": cands, "best_iou": best})
        print(f"features {name}: {len(cands)} candidates", flush=True)

    X = np.vstack([matrix(im["cands"]) for im in images])
    y = np.concatenate([(im["best_iou"] >= 0.5).astype(int) for im in images])
    groups = np.concatenate([[k] * len(im["cands"]) for k, im in enumerate(images)])
    p = out_of_fold(X, y, groups)

    # Ablations: which signals carry the verifier?
    keep_no_sam = [i for i, f in enumerate(FEATURES) if f not in SAM_SIGNALS]
    keep_sam = [i for i, f in enumerate(FEATURES) if f in SAM_SIGNALS]
    p_no_sam = out_of_fold(X[:, keep_no_sam], y, groups)
    p_sam_only = out_of_fold(X[:, keep_sam], y, groups)
    has_conf = X[:, FEATURES.index("sam_conf_missing")] == 0
    detection = {
        "SAM 3.1 confidence alone": auroc(X[has_conf, FEATURES.index("sam_conf")], y[has_conf]),
        "Model agreement alone": auroc(X[:, FEATURES.index("cross_iou")], y),
        "Verifier: shape + appearance only": auroc(p_no_sam, y),
        "Verifier: SAM signals only": auroc(p_sam_only, y),
        "VeriCell verifier (all signals)": auroc(p, y),
    }
    ece_val, reliability = ece(p, y)

    # Fusion, review queue, review-efficiency curve, counts.
    totals = {k: [0, 0, 0] for k in ("cellpose", "sam_tiled", "vericell")}
    curve_cells = []
    counts_pred, counts_true, per_image, offset = [], [], [], 0
    for k, im in enumerate(images):
        cands, gt = im["cands"], im["gt"]
        pk = p[offset:offset + len(cands)]
        offset += len(cands)
        accepted = fuse(cands, pk)
        acc_set = set(accepted.tolist())
        acc_masks = [cands[i].inst.mask for i in accepted]
        # A rejected cell is a "possible missing cell" if it doesn't overlap anything accepted.
        free = []
        if acc_masks:
            ov = iou_matrix([c.inst.mask for c in cands], acc_masks).max(1)
        else:
            ov = np.zeros(len(cands))
        for i in range(len(cands)):
            if i not in acc_set and ov[i] < 0.2:
                free.append(i)
        review = [i for i in list(accepted) + free if REVIEW_BAND[0] <= pk[i] < REVIEW_BAND[1]]

        for key, masks in (("cellpose", [c.mask for c in im["cp"]]), ("sam_tiled", [c.mask for c in im["tx"]]),
                           ("vericell", acc_masks)):
            totals[key][0] += len(masks); totals[key][1] += len(gt); totals[key][2] += score(masks, gt)

        # Review-efficiency: candidates whose accept/reject decision a reviewer could flip.
        for i in list(accepted) + free:
            curve_cells.append({"img": k, "i": i, "p": float(pk[i]), "accepted": i in acc_set,
                                "correct": bool(im["best_iou"][i] >= 0.5)})

        n_hat = float(pk[accepted].sum() + pk[free].sum())
        counts_pred.append(n_hat); counts_true.append(len(gt))
        im.update(p=pk, accepted=acc_set, free=set(free), review=set(review))
        per_image.append({"stem": im["stem"], "cell_type": im["meta"]["cell_type"], "n_gt": len(gt),
                          "n_cellpose": len(im["cp"]), "n_accepted": len(accepted), "n_review": len(review),
                          "f1_cellpose": prf(len(im["cp"]), len(gt), score([c.mask for c in im["cp"]], gt))["f1"],
                          "f1_vericell": prf(len(accepted), len(gt), score(acc_masks, gt))["f1"],
                          "count_estimate": n_hat})

    lo, hi = conformal_counts(np.array(counts_pred), np.array(counts_true, dtype=float))
    for row, a, b in zip(per_image, lo, hi):
        row["count_lo"], row["count_hi"] = float(a), float(b)
    covered = [(a <= t <= b) for a, b, t in zip(lo, hi, counts_true)]

    # Review-efficiency: TP gained per reviewed cell, uncertain-first vs. random order.
    def curve(order):
        cells = [curve_cells[i] for i in order]
        n_gt = totals["vericell"][1]
        tp, npred = totals["vericell"][2], totals["vericell"][0]
        pts = [{"reviewed": 0, "f1": 2 * tp / (npred + n_gt)}]
        for j, c in enumerate(cells, 1):
            # Reviewer fixes the decision: wrongly accepted -> removed; correct but rejected -> added.
            if c["accepted"] and not c["correct"]:
                npred -= 1
            elif not c["accepted"] and c["correct"]:
                npred += 1; tp += 1
            if j % max(1, len(cells) // 40) == 0 or j == len(cells):
                pts.append({"reviewed": j / len(cells), "f1": 2 * tp / (npred + n_gt)})
        return pts

    uncertain_first = sorted(range(len(curve_cells)), key=lambda i: abs(curve_cells[i]["p"] - 0.5))
    rng = np.random.default_rng(0)
    review_curves = {"uncertain first (VeriCell)": curve(uncertain_first),
                     "random order": curve(list(rng.permutation(len(curve_cells))))}

    cp_count_err = np.median([abs(r["n_cellpose"] - r["n_gt"]) / r["n_gt"] for r in per_image])
    vc_count_err = np.median([abs(r["count_estimate"] - r["n_gt"]) / r["n_gt"] for r in per_image])
    summary = {
        "dataset": f"LIVECell test sample: {len(images)} images, 8 cell types, {totals['vericell'][1]} cells",
        "segmentation": {
            "Cellpose-SAM": prf(totals["cellpose"][0], totals["cellpose"][1], totals["cellpose"][2]),
            "SAM 3.1 text, zoomed tiles": prf(totals["sam_tiled"][0], totals["sam_tiled"][1], totals["sam_tiled"][2]),
            "VeriCell (verified fusion)": prf(totals["vericell"][0], totals["vericell"][1], totals["vericell"][2]),
        },
        "error_detection_auroc": detection,
        "calibration": {"ece": ece_val, "reliability": reliability},
        "review_curves": review_curves,
        "counts": {"coverage_90": float(np.mean(covered)),
                   "median_rel_width": float(np.median((hi - lo) / np.array(counts_true))),
                   "cellpose_median_abs_error": float(cp_count_err),
                   "vericell_median_abs_error": float(vc_count_err)},
        "per_image": per_image,
        "n_candidates": int(len(y)), "frac_correct_candidates": float(y.mean()),
    }

    # ---- Export for the UI ----
    (UI / "img").mkdir(parents=True, exist_ok=True)
    (UI / "cells").mkdir(parents=True, exist_ok=True)
    for im in images:
        g = np.asarray(im["image"].convert("L"), dtype=np.float32)
        lo_, hi_ = np.percentile(g, (0.5, 99.5))
        cv2.imwrite(str(UI / "img" / f"{im['stem']}.png"), np.clip((g - lo_) / (hi_ - lo_ + 1e-6) * 255, 0, 255).astype(np.uint8))
        cells = []
        for i, c in enumerate(im["cands"]):
            status = "accepted" if i in im["accepted"] else ("possible_missing" if i in im["free"] else "suppressed")
            cells.append({"id": i, "source": c.source, "p": round(float(im["p"][i]), 3), "status": status,
                          "review": i in im["review"], "correct": bool(im["best_iou"][i] >= 0.5),
                          "best_iou": round(float(im["best_iou"][i]), 3), "box": list(c.inst.box),
                          "poly": polygon(c.inst.mask), "reasons": reasons(c.features),
                          "features": {k: round(v, 3) for k, v in c.features.items()}})
        (UI / "cells" / f"{im['stem']}.json").write_text(json.dumps({
            "stem": im["stem"], "cell_type": im["meta"]["cell_type"], "width": im["image"].width,
            "height": im["image"].height, "cells": cells, "gt": [polygon(m) for m in im["gt"] if m.any()],
            "stats": next(r for r in per_image if r["stem"] == im["stem"])}))
    (UI / "summary.json").write_text(json.dumps(summary, indent=1))

    print("\n== Segmentation (F1 @ IoU 0.5, out-of-fold) ==")
    for k, v in summary["segmentation"].items():
        print(f"  {k:30} F1 {v['f1']:.3f}  precision {v['precision']:.3f}  recall {v['recall']:.3f}")
    print("\n== Can it tell right cells from wrong ones? (AUROC) ==")
    for k, v in detection.items():
        print(f"  {k:36} {v:.3f}")
    print(f"\nCalibration ECE {ece_val:.3f}")
    c = summary["counts"]
    print(f"Counts: 90% intervals cover {c['coverage_90']:.0%} of images, median width ±{c['median_rel_width'] / 2:.0%}; "
          f"median count error Cellpose {c['cellpose_median_abs_error']:.1%} vs VeriCell {c['vericell_median_abs_error']:.1%}")
    rc = review_curves["uncertain first (VeriCell)"]
    at10 = next(pt for pt in rc if pt["reviewed"] >= 0.1)
    print(f"Review: checking the {at10['reviewed']:.0%} most uncertain cells lifts F1 to {at10['f1']:.3f}")


if __name__ == "__main__":
    main()
