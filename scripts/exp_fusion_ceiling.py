"""How much could per-cell fusion of Cellpose + tiled SAM text gain? (cached predictions, no API calls)

Reports, on the zoom-experiment images:
  - naive union: Cellpose plus every SAM cell that overlaps no Cellpose cell
  - oracle ceilings: what a perfect per-cell verifier could reach
  - a taxonomy of Cellpose's errors (what the repair agent would have to fix)
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from scripts.exp_zoom import PICK  # noqa: E402
from vericell import data  # noqa: E402
from vericell.metrics import iou_matrix, match  # noqa: E402


def f1(tp, npred, ngt):
    return 2 * tp / (npred + ngt)


def main() -> None:
    agg = Counter()
    taxonomy = Counter()
    ious = {"cellpose": [], "oracle_best_mask": []}
    for name, _img, gt, _meta in data.livecell():
        if name not in PICK:
            continue
        st = name[:-4]
        cp = load(ROOT / f"outputs/cache/{st}.cellpose.json")
        tx = load(ROOT / f"outputs/cache/{st}.sam_text_tiled.json")
        hz = load(ROOT / f"outputs/cache/{st}.hybrid_zoom.json")
        n = len(gt)
        agg["gt"] += n

        iou_cp = iou_matrix([i.mask for i in cp], gt)
        best_cp, pairs_cp = match(iou_cp)
        agg["cp_pred"] += len(cp); agg["cp_tp"] += len(pairs_cp)
        ious["cellpose"] += [iou_cp[r, c] for r, c in pairs_cp]

        # Naive union: add SAM cells that don't overlap any Cellpose cell (IoU < 0.2).
        cross = iou_matrix([i.mask for i in tx], [i.mask for i in cp])
        new = [t for t, row in zip(tx, cross) if (row.max() if row.size else 0) < 0.2]
        union = cp + new
        _, pu = match(iou_matrix([i.mask for i in union], gt))
        agg["union_pred"] += len(union); agg["union_tp"] += len(pu)

        # Oracle A: perfect verifier only deletes wrong Cellpose cells.
        agg["oA_pred"] += len(pairs_cp); agg["oA_tp"] += len(pairs_cp)
        # Oracle B: perfect verifier over all candidates: keeps one correct mask per GT cell if any exists.
        cands = cp + tx + hz
        iou_all = iou_matrix([i.mask for i in cands], gt)
        best_per_gt = iou_all.max(0) if len(cands) else np.zeros(n)
        covered = best_per_gt >= 0.5
        agg["oB_tp"] += int(covered.sum())
        ious["oracle_best_mask"] += list(best_per_gt[covered])

        # Taxonomy of Cellpose errors.
        matched_gt = {c for _, c in pairs_cp}
        for r in range(len(cp)):
            if best_cp[r] >= 0.5:
                continue
            row = iou_cp[r]
            inter_frac = [(cp[r].mask & gt[c]).sum() / gt[c].sum() for c in np.nonzero(row > 0)[0]]
            if sum(f > 0.5 for f in inter_frac) >= 2:
                taxonomy["FP: merge (covers 2+ cells)"] += 1
            elif row.max() >= 0.2:
                taxonomy["FP: poor boundary / fragment (IoU 0.2-0.5)"] += 1
            else:
                taxonomy["FP: spurious (no real cell)"] += 1
        for c in range(n):
            if c in matched_gt:
                continue
            col = iou_cp[:, c] if len(cp) else np.zeros(1)
            if col.max() >= 0.2:
                taxonomy["FN: found but badly drawn"] += 1
            elif col.max() > 0:
                taxonomy["FN: part of a merge / touched only"] += 1
            else:
                taxonomy["FN: completely missed"] += 1
            if covered[c]:
                taxonomy["  of FNs, rescuable by another candidate"] += 1

    g = agg["gt"]
    print(f"{len(PICK)} images, {g} GT cells (F1 @ IoU 0.5)\n")
    print(f"Cellpose-SAM alone            F1 {f1(agg['cp_tp'], agg['cp_pred'], g):.3f}  recall {agg['cp_tp'] / g:.3f}")
    print(f"Naive union (+SAM-only cells) F1 {f1(agg['union_tp'], agg['union_pred'], g):.3f}  recall {agg['union_tp'] / g:.3f}  precision {agg['union_tp'] / agg['union_pred']:.3f}")
    print(f"Oracle A: drop bad Cellpose   F1 {f1(agg['oA_tp'], agg['oA_pred'], g):.3f}  (verifier only, no repair)")
    print(f"Oracle B: best of all cands   F1 {f1(agg['oB_tp'], agg['oB_tp'], g):.3f}  recall {agg['oB_tp'] / g:.3f}  (verifier + pick + rescue)")
    print(f"Mask IoU on matched cells: Cellpose {np.mean(ious['cellpose']):.3f} -> best-candidate oracle {np.mean(ious['oracle_best_mask']):.3f}")
    print("\nCellpose error taxonomy:")
    for k, v in sorted(taxonomy.items(), key=lambda kv: kv[0]):
        print(f"  {k:45} {v}")


if __name__ == "__main__":
    main()
