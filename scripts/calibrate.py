"""Calibrate the agent's 90% answer ranges (split conformal), per question type and domain.

For every labelled image, candidates are scored by a verifier that never saw that image
(out-of-fold), answers are computed exactly as the agent does, and each answer's error is
normalised by its Monte Carlo half-width (see vericell/answers.py). The 90% conformal quantile of
those scores is saved to outputs/models/{domain}_calibration.json, and leave-one-image-out coverage
is reported so the "90%" claim is checked, not assumed.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from vericell import answers, api, data, pipeline  # noqa: E402
from vericell.features import build_candidates  # noqa: E402
from vericell.measure import answer, measure_masks, tertile_means  # noqa: E402
from vericell.metrics import iou_matrix  # noqa: E402
from vericell.sam import SAM  # noqa: E402
from vericell.verifier import matrix, out_of_fold  # noqa: E402

MODELS = ROOT / "outputs" / "models"
METRICS = ["Area_outer", "Perimeter_outer", "roundness", "perimeter_diff"]
ALPHA = 0.1


def plans() -> list[dict]:
    out = [{"metric": "count", "operation": "count", "fields": ["count"]}]
    for m in METRICS:
        out.append({"metric": m, "operation": "max", "fields": [m, "x", "y"] if m == "Area_outer" else [m]})
        out.append({"metric": m, "operation": "tertile_mean", "fields": [m]})
        out.append({"metric": m, "operation": "mean", "fields": [m]})
    return out


def truth(plan: dict, gt_table: list[dict]) -> dict:
    if plan["operation"] == "count":
        return {"count": float(len(gt_table))}
    if plan["operation"] == "tertile_mean":
        return tertile_means(np.array([r[plan["metric"]] for r in gt_table]))
    return answer(gt_table, plan["metric"], plan["operation"], plan["fields"])


def conformal_q(scores: list[float]) -> float:
    s = np.sort(np.asarray(scores))
    k = int(np.ceil((len(s) + 1) * (1 - ALPHA))) - 1
    return float(s[min(k, len(s) - 1)])


def calibrate(domain: str, images: list) -> None:
    """images: list of (name, PIL image, candidates, gt masks)."""
    X = np.vstack([matrix(c) for _, _, c, _ in images])
    y = np.concatenate([(iou_matrix([x.inst.mask for x in c], gt).max(1) >= 0.5).astype(int) for _, _, c, gt in images])
    groups = np.concatenate([np.full(len(c), k) for k, (_, _, c, _) in enumerate(images)])
    p_all = out_of_fold(X, y, groups)

    scores = defaultdict(list)  # key -> list of (image index, score)
    offset = 0
    for k, (name, _img, cands, gt) in enumerate(images):
        p = p_all[offset:offset + len(cands)]
        offset += len(cands)
        table = [(measure_masks([c.inst.mask]) or [None])[0] for c in cands]
        idx = answers.pool(cands, p, table)
        gt_table = measure_masks(gt)
        for plan in plans():
            val, _ = answers.point(plan, table, p, idx)
            if not val or not gt_table:
                continue
            hw = answers.mc_halfwidth(plan, table, p, idx)
            ref = truth(plan, gt_table)
            for f, v in val.items():
                if v is None or ref.get(f) is None:
                    continue
                d = {kind: answers.halfwidth(kind, v, hw.get(f, 0.0)) for kind in ("mc", "rel")}
                if min(d.values()) > 0:
                    scores[answers.calibration_key(plan, f)].append(
                        (k, {kind: abs(ref[f] - v) / d[kind] for kind in d}, d))
        print(f"{domain} {name}: {len(cands)} candidates", flush=True)

    calib, report = {}, {}
    for key, rows in scores.items():
        best = None
        for kind in ("mc", "rel"):
            s = [x[kind] for _, x, _ in rows]
            # Leave-one-image-out: is the truth inside the range built without that image, and how wide is it?
            loo = [(x[kind] <= (qi := conformal_q([y[kind] for j, y, _ in rows if j != i])), qi * d[kind])
                   for i, x, d in rows]
            cov = float(np.mean([h for h, _ in loo]))
            width = float(np.mean([w for _, w in loo]))  # mean, not median: penalise ranges that blow up on hard images
            cand = {"q": conformal_q(s), "kind": kind, "loo_coverage": round(cov, 3), "mean_halfwidth": width}
            if best is None or width < best["mean_halfwidth"]:
                best = cand
        calib[key] = {"q": best["q"], "kind": best["kind"]}
        report[key] = {"q": round(best["q"], 3), "kind": best["kind"], "n": len(rows),
                       "loo_coverage": best["loo_coverage"]}
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / f"{domain}_calibration.json").write_text(json.dumps({"q": calib, "report": report}, indent=1))
    print(f"\n{domain}: leave-one-image-out coverage of 90% ranges")
    for key, r in sorted(report.items()):
        print(f"  {key:40} {r['kind']:3} q={r['q']:7.3f}  coverage {r['loo_coverage']:.0%}  (n={r['n']})")


def organoids() -> None:
    client = api.client()
    sam = SAM(client)
    D, cache = ROOT / "data" / "morphoorga", ROOT / "outputs" / "morphoorga" / "cache"
    images = []
    for img_path in sorted((D / "images").glob("*.png")):
        n = img_path.stem
        image = Image.open(img_path).convert("RGB")
        props = pipeline.propose(client, sam, image, name=n, cache=cache)
        cands = pipeline.candidates(image, props)
        merged = [i for i, c in enumerate(cands) if c.source == "merged"]
        pipeline.judge_candidates(client, image, cands, merged, props.scene["target_object"], name=n, cache=cache)
        lab = np.load(D / "masks" / f"{n}.npy")
        images.append((n, image, cands, [lab == i for i in np.unique(lab) if i > 0]))
    calibrate("organoid", images)


def cells() -> None:
    cache = ROOT / "outputs" / "cache"
    images = []
    for name, image, gt, _meta in data.livecell():
        stem = name[:-4]
        cp, tx, zm = (load(cache / f"{stem}.{m}.json") for m in ("cellpose", "sam_text_tiled", "hybrid_zoom"))
        images.append((stem, image, build_candidates(image, cp, tx, zm), gt))
    calibrate("cell", images)


if __name__ == "__main__":
    which = sys.argv[1:] or ["organoid", "cell"]
    if "organoid" in which:
        organoids()
    if "cell" in which:
        cells()
