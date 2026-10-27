"""Full MorphoOrgaVQA run: plan -> segment -> verify -> measure -> answer, scored by the official scorer.

Per image (cached in outputs/morphoorga/cache):
  scene     Muse Spark names the objects to segment, as SAM-friendly noun phrases
  cellpose  Cellpose-SAM instances
  sam_text  SAM 3.1 text prompts (the scene's top 2 phrases) on zoomed tiles, de-duplicated
  zoom      SAM 3.1 zoomed re-segmentation of every Cellpose box (confidence + second opinion)
Then the per-cell verifier (out-of-fold over images) scores every candidate and fusion picks the
final organoids. Per question, the planner's JSON plan runs on the measurement engine. Ground truth
and scoring follow the benchmark's published protocol (vericell/vqa_score.py).

Methods compared (all through the same planner + measurement engine):
  sam_text   SAM 3.1 text prompt only
  cellpose   Cellpose-SAM only
  vericell   verified fusion
  gt_masks   ground-truth masks (sanity check: should score ~0% error)

Usage:
    .venv/bin/python scripts/get_morphoorga.py
    .venv/bin/python scripts/run_morphoorga.py
"""

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load, save  # noqa: E402
from vericell import api, proposers  # noqa: E402
from vericell.features import build_candidates  # noqa: E402
from vericell.measure import answer, measure_masks  # noqa: E402
from vericell.metrics import iou_matrix, match  # noqa: E402
from vericell.planner import Planner, describe_image  # noqa: E402
from vericell.sam import SAM, _dedupe  # noqa: E402
from vericell.verifier import fuse, matrix, out_of_fold  # noqa: E402
from vericell.vqa_score import groundtruth, score  # noqa: E402

DATA = ROOT / "data" / "morphoorga"
OUT = ROOT / "outputs" / "morphoorga"
CACHE = OUT / "cache"
CATEGORY_METRIC = {"Area": "Area_outer", "Perimeter": "Perimeter_outer", "Roundness": "roundness",
                   "Roughness": "perimeter_diff"}
# MorphoOrgaAgent, arXiv 2609.08696 Table 1: median APE (%), (max clear, max open, tertile clear, tertile open)
PUBLISHED = {"Area": (16.77, 17.40, 38.74, 36.77), "Perimeter": (15.36, 15.36, 22.03, 22.24),
             "Roughness": (46.02, 46.02, 31.75, 33.74), "Roundness": (1.54, 1.56, 2.48, 2.51)}


def tiling(image: Image.Image) -> tuple[int, float]:
    """Tiles per side and upscale so each tile reaches SAM's ~1008 px working size."""
    side = max(image.size)
    tiles = int(np.clip(round(side / 700), 1, 4))
    return tiles, float(np.clip(1008 / (side / tiles), 1.0, 3.0))


def cached_instances(path: Path, fn):
    if path.exists():
        return load(path)
    inst = fn()
    save(path, inst)
    return inst


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    client = api.client()
    sam, planner = SAM(client), Planner(client)

    names = sorted(p.stem for p in (DATA / "images").glob("*.png"))
    images = {n: Image.open(DATA / "images" / f"{n}.png").convert("RGB") for n in names}

    # 1. Scene descriptions (one Spark look per image).
    def scene(n):
        path = CACHE / f"{n}.scene.json"
        if not path.exists():
            path.write_text(json.dumps(describe_image(client, images[n])))
        return n, json.loads(path.read_text())
    with ThreadPoolExecutor(6) as ex:
        scenes = dict(ex.map(scene, names))

    # 2. Candidates. Cellpose runs locally, SAM calls in parallel.
    cp = {n: cached_instances(CACHE / f"{n}.cellpose.json", lambda n=n: proposers.cellpose(images[n])) for n in names}
    print("cellpose done", flush=True)

    def sam_sources(n):
        tiles, scale = tiling(images[n])
        def text():
            found = []
            for concept in scenes[n]["sam_concepts"][:2]:
                found += sam.segment_text_tiled(images[n], concept, tiles=tiles, scale=scale)
            return _dedupe(found)
        t = cached_instances(CACHE / f"{n}.sam_text.json", text)
        z = cached_instances(CACHE / f"{n}.zoom.json", lambda: sam.segment_boxes_zoomed(images[n], [i.box for i in cp[n]]))
        print(f"{n}: cellpose {len(cp[n])}, sam {len(t)}, zoom {len(z)}", flush=True)
        return n, t, z
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(sam_sources, names))
    tx = {n: t for n, t, _ in res}
    zm = {n: z for n, _, z in res}

    # 3. Verifier, out-of-fold over images.
    gt_masks = {}
    cands, best = {}, {}
    for n in names:
        lab = np.load(DATA / "masks" / f"{n}.npy")
        gt_masks[n] = [lab == i for i in np.unique(lab) if i > 0]
        cands[n] = build_candidates(images[n], cp[n], tx[n], zm[n])
        best[n] = iou_matrix([c.inst.mask for c in cands[n]], gt_masks[n]).max(1) if cands[n] else np.zeros(0)
    X = np.vstack([matrix(cands[n]) for n in names if cands[n]])
    y = np.concatenate([(best[n] >= 0.5).astype(int) for n in names if cands[n]])
    groups = np.concatenate([[k] * len(cands[n]) for k, n in enumerate(names) if cands[n]])
    p_all = out_of_fold(X, y, groups)
    fused, offset = {}, 0
    for n in names:
        k = len(cands[n])
        p = p_all[offset:offset + k]
        offset += k
        fused[n] = [cands[n][i].inst.mask for i in fuse(cands[n], p)] if k else []

    methods = {
        "sam_text": {n: [i.mask for i in tx[n]] for n in names},
        "cellpose": {n: [i.mask for i in cp[n]] for n in names},
        "vericell": fused,
        "gt_masks": gt_masks,
    }

    # Segmentation quality per method.
    seg = {}
    for m, masks in methods.items():
        tp = npred = ngt = 0
        for n in names:
            tp += len(match(iou_matrix(masks[n], gt_masks[n]))[1]); npred += len(masks[n]); ngt += len(gt_masks[n])
        seg[m] = {"precision": tp / max(npred, 1), "recall": tp / ngt, "f1": 2 * tp / (npred + ngt)}

    # 4. Plans (one per unique question) -> answers -> official scorer.
    questions = json.loads((DATA / "benchmark_questions.json").read_text())
    slots = [(c, mode, slot, q) for c in questions for mode in ("clear", "open")
             for slot, q in zip(("max", "tertile_mean"), questions[c][mode])]
    with ThreadPoolExecutor(8) as ex:
        plans = dict(zip([s[3] for s in slots], ex.map(planner.plan, [s[3] for s in slots])))
    (OUT / "plans.json").write_text(json.dumps(plans, indent=1))
    planner_ok = sum(plans[q]["metric"] == CATEGORY_METRIC[c] and plans[q]["operation"] == slot for c, _, slot, q in slots)

    gts = {f"{n}.png": groundtruth(np.load(DATA / "masks" / f"{n}.npy")) for n in names}
    results = {}
    for m, masks in methods.items():
        tables = {n: measure_masks(masks[n]) for n in names}
        preds = defaultdict(lambda: defaultdict(dict))
        for n in names:
            for c, mode, slot, q in slots:
                plan = plans[q]
                ans = answer(tables[n], plan["metric"], plan["operation"], plan["fields"])
                preds[f"{n}.png"][f"{c}__{slot}"][mode] = {"final_answer": ans}
        records = score(preds, gts)
        (OUT / f"records_{m}.json").write_text(json.dumps(records))
        table = {}
        for c in CATEGORY_METRIC:
            row = []
            for slot in ("max", "tertile_mean"):
                for mode in ("clear", "open"):
                    errs = [r["pct_error"] for r in records if r["category"] == c and r["slot_type"] == slot
                            and r["mode"] == mode and r["pct_error"] is not None]
                    row.append(float(np.median(errs)) if errs else float("nan"))
            table[c] = row
        credit = np.mean([r["credit"] for r in records])
        results[m] = {"median_ape": table, "mean_credit": float(credit), "segmentation": seg[m]}

    (OUT / "results.json").write_text(json.dumps({"results": results, "published_morphoorgaagent": PUBLISHED,
                                                  "planner_correct": planner_ok, "n_questions": len(slots),
                                                  "n_images": len(names)}, indent=1))

    print(f"\nMorphoOrgaVQA: {len(names)} images x {len(slots)} questions. Planner picked the right metric + "
          f"operation for {planner_ok}/{len(slots)} questions.\n")
    print("Median APE (%)  [max clear | max open | tertile clear | tertile open]")
    for c in CATEGORY_METRIC:
        print(f"  {c:10} MorphoOrgaAgent " + " | ".join(f"{v:6.2f}" for v in PUBLISHED[c]))
        for m in methods:
            print(f"  {'':10} {m:15} " + " | ".join(f"{v:6.2f}" for v in results[m]["median_ape"][c]))
    print("\nMean credit (1 = within 15%, 0.5 = within 30%) and segmentation F1:")
    for m in methods:
        s = results[m]["segmentation"]
        print(f"  {m:10} credit {results[m]['mean_credit']:.3f}   F1 {s['f1']:.3f} (P {s['precision']:.3f}, R {s['recall']:.3f})")


if __name__ == "__main__":
    main()
