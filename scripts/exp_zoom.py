"""Experiment: does zooming in fix SAM 3.1's dense-scene recall and small-cell edges?

Compares, on a few LIVECell images, the cached baselines against
  - sam_text_tiled: text prompt on a 3x3 grid of 3x-upscaled tiles
  - hybrid_zoom:    Cellpose boxes re-segmented by SAM 3.1 on 3x-upscaled windows
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import cached  # noqa: E402
from vericell import api, data  # noqa: E402
from vericell.metrics import iou_matrix, match, scores  # noqa: E402
from vericell.sam import SAM  # noqa: E402

PICK = ["MCF7_Phase_H4_1_03d00h00m_1.tif", "SHSY5Y_Phase_A10_2_02d20h00m_3.tif", "BV2_Phase_A4_2_01d20h00m_3.tif",
        "A172_Phase_C7_2_01d04h00m_4.tif", "SkBr3_Phase_G3_2_02d16h00m_3.tif", "Huh7_Phase_A12_1_03d16h00m_4.tif"]


def main() -> None:
    sam = SAM(api.client())
    samples = [s for s in data.livecell() if s[0] in PICK]

    def run(sample):
        name, image, gt, meta = sample
        cp = cached(name, "cellpose", lambda: None)
        preds = {
            "sam_text_cell": cached(name, "sam_text_cell", lambda: None),
            "sam_text_tiled": cached(name, "sam_text_tiled", lambda: sam.segment_text_tiled(image, "cell")),
            "cellpose": cp,
            "hybrid": cached(name, "hybrid", lambda: None),
            "hybrid_zoom": cached(name, "hybrid_zoom", lambda: sam.segment_boxes_zoomed(image, [i.box for i in cp])),
        }
        rows = {}
        for m, inst in preds.items():
            iou = iou_matrix([i.mask for i in inst], gt)
            best, pairs = match(iou)
            s = scores(len(inst), len(gt), pairs, iou)
            small = [iou[r, c] for r, c in pairs if gt[c].sum() < 200]
            s["small_iou"] = float(np.mean(small)) if small else float("nan")
            rows[m] = s
        print(f"done {name}", flush=True)
        return name, rows

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, samples))

    methods = list(results[0][1])
    print(f"\n{'image':36} " + " ".join(f"{m:>15}" for m in methods) + "   (F1 @0.5)")
    for name, rows in results:
        print(f"{name[:36]:36} " + " ".join(f"{rows[m]['f1']:15.3f}" for m in methods))
    print()
    for m in methods:
        tp = sum(r[m]["tp"] for _, r in results); npred = sum(r[m]["n_pred"] for _, r in results)
        ngt = sum(r[m]["n_gt"] for _, r in results)
        miou = sum(r[m]["mean_matched_iou"] * r[m]["tp"] for _, r in results) / max(tp, 1)
        small = np.nanmean([r[m]["small_iou"] for _, r in results])
        print(f"{m:16} F1 {2 * tp / (npred + ngt):.3f}  prec {tp / npred:.3f}  recall {tp / ngt:.3f}  "
              f"mIoU {miou:.3f}  small-cell IoU {small:.3f}")


if __name__ == "__main__":
    main()
