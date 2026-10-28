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
from vericell.features import add_judgement, build_candidates  # noqa: E402
from vericell.judge import judge  # noqa: E402
from vericell.measure import answer, answer_soft, drop_small, measure_masks  # noqa: E402
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
STRICT_EXTREME_P = 0.7  # fixed in advance, not tuned on the benchmark
DEBRIS_MIN_AREA = 200  # OrganoID (Matthews et al. 2022): "discarded if the total area was below 200 pixels"
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

    # 2. Candidates. Cellpose runs locally (native + coarser pyramid levels), SAM calls in parallel.
    cp, coarse, cp_ms = {}, {}, {}
    for n in names:
        cp[n] = cached_instances(CACHE / f"{n}.cellpose.json", lambda n=n: proposers.cellpose(images[n]))
        by_scale = {1.0: cp[n]}
        for s in proposers.pyramid_scales(images[n])[1:]:
            by_scale[s] = cached_instances(CACHE / f"{n}.cellpose_s{s:.3f}.json",
                                           lambda n=n, s=s: proposers.cellpose_scaled(images[n], s))
        coarse[n] = [i for s, insts in by_scale.items() if s < 1.0 for i in insts]
        cp_ms[n] = proposers.merge_scales(by_scale)
    print("cellpose pyramid done", flush=True)

    def sam_sources(n):
        tiles, scale = tiling(images[n])
        def text():
            found = []
            for concept in scenes[n]["sam_concepts"][:2]:
                found += sam.segment_text_tiled(images[n], concept, tiles=tiles, scale=scale)
            return _dedupe(found)
        t = cached_instances(CACHE / f"{n}.sam_text.json", text)
        z = cached_instances(CACHE / f"{n}.zoom.json", lambda: sam.segment_boxes_zoomed(images[n], [i.box for i in cp[n]]))
        print(f"{n}: cellpose {len(cp[n])} (+{len(coarse[n])} coarse), sam {len(t)}, zoom {len(z)}", flush=True)
        return n, t, z
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(sam_sources, names))
    tx = {n: t for n, t, _ in res}
    zm = {n: z for n, _, z in res}

    # 3. Verifier, out-of-fold over images. v1 = Cellpose + SAM; v2 adds the Cellpose pyramid and
    #    merged touching parts as candidates, and answers tier questions with probability weights.
    gt_masks = {}
    for n in names:
        lab = np.load(DATA / "masks" / f"{n}.npy")
        gt_masks[n] = [lab == i for i in np.unique(lab) if i > 0]

    def judged(n, cands_n):
        """Ask Muse Spark about every merged candidate (cached per image)."""
        path = CACHE / f"{n}.judge.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        todo = [c for c in cands_n if c.source == "merged" and str(list(c.inst.box)) not in cache]
        with ThreadPoolExecutor(8) as ex:
            for c, v in zip(todo, ex.map(lambda c: judge(client, images[n], c.inst, scenes[n]["target_object"]), todo)):
                cache[str(list(c.inst.box))] = v
        if todo:
            path.write_text(json.dumps(cache))
        for c in cands_n:
            if c.source == "merged":
                add_judgement(c, cache[str(list(c.inst.box))])

    def verify(extra_fn, use_judge=False):
        cands = {n: build_candidates(images[n], cp[n], tx[n], zm[n], extra=extra_fn(n)) for n in names}
        if use_judge:
            for n in names:
                judged(n, cands[n])
            print("crop judge done", flush=True)
        best = {n: iou_matrix([c.inst.mask for c in cands[n]], gt_masks[n]).max(1) if cands[n] else np.zeros(0)
                for n in names}
        X = np.vstack([matrix(cands[n]) for n in names if cands[n]])
        y = np.concatenate([(best[n] >= 0.5).astype(int) for n in names if cands[n]])
        groups = np.concatenate([[k] * len(cands[n]) for k, n in enumerate(names) if cands[n]])
        p_all = out_of_fold(X, y, groups)
        hard, soft, strict, offset = {}, {}, {}, 0
        for n in names:
            k = len(cands[n])
            p = p_all[offset:offset + k]
            offset += k
            acc = fuse(cands[n], p) if k else []
            hard[n] = [cands[n][i].inst.mask for i in acc]
            # "Most extreme object" answers hinge on one object, so only confident ones may win.
            strict[n] = [cands[n][i].inst.mask for i in acc if p[i] >= STRICT_EXTREME_P]
            keep = fuse(cands[n], p, threshold=0.05) if k else []   # every plausible, non-duplicate object
            soft[n] = ([cands[n][i].inst.mask for i in keep], [float(p[i]) for i in keep])
        print(f"verifier done: {len(y)} candidates", flush=True)
        return hard, soft, strict

    v1_hard, _, _ = verify(lambda n: None)
    pool = lambda n: {"coarse": coarse[n], "merged": proposers.touching_clusters(cp[n])}  # noqa: E731
    v2_hard, v2_soft, _ = verify(pool)
    v3_hard, v3_soft, v3_strict = verify(pool, use_judge=True)

    # method -> (masks, optional (masks, weights) for probability-weighted tiers, optional masks for extremes)
    methods = {
        "sam_text": ({n: [i.mask for i in tx[n]] for n in names}, None, None),
        "cellpose": ({n: [i.mask for i in cp[n]] for n in names}, None, None),
        "cellpose_ms": ({n: [i.mask for i in cp_ms[n]] for n in names}, None, None),
        "vericell_v1": (v1_hard, None, None),
        "vericell_v2": (v2_hard, v2_soft, None),
        "vericell_v3": (v3_hard, v3_soft, None),
        "vericell_v3_strict": (v3_hard, v3_soft, v3_strict),
        "gt_masks": (gt_masks, None, None),
    }

    # 4. Plans (one per unique question) -> answers -> scoring.
    questions = json.loads((DATA / "benchmark_questions.json").read_text())
    slots = [(c, mode, slot, q) for c in questions for mode in ("clear", "open")
             for slot, q in zip(("max", "tertile_mean"), questions[c][mode])]
    with ThreadPoolExecutor(8) as ex:
        plans = dict(zip([s[3] for s in slots], ex.map(planner.plan, [s[3] for s in slots])))
    (OUT / "plans.json").write_text(json.dumps(plans, indent=1))
    planner_ok = sum(plans[q]["metric"] == CATEGORY_METRIC[c] and plans[q]["operation"] == slot for c, _, slot, q in slots)

    # Two ground-truth regimes. "official": the benchmark as published. "debris_filtered": OrganoID's
    # own rule (its paper discards objects under 200 px) applied to OrganoID images, to ground truth
    # and predictions alike; the benchmark export left those specks in.
    def min_area(n, regime):
        return DEBRIS_MIN_AREA if regime == "debris_filtered" and n.startswith("organoid_") else 0

    def evaluate(masks, soft, extreme, regime):
        gts, preds = {}, defaultdict(lambda: defaultdict(dict))
        tp = npred = ngt = 0
        for n in names:
            lab = np.load(DATA / "masks" / f"{n}.npy")
            a = min_area(n, regime)
            if a:
                lab = drop_small(lab, a)
            gts[f"{n}.png"] = groundtruth(lab)
            gt_n = [g for g in gt_masks[n] if g.sum() >= a]
            ms = [m for m in masks[n] if m.sum() >= a]
            tp += len(match(iou_matrix(ms, gt_n))[1]); npred += len(ms); ngt += len(gt_n)
            table = measure_masks(ms)
            ext_table = measure_masks([m for m in extreme[n] if m.sum() >= a]) if extreme is not None else table
            if not ext_table:
                ext_table = table
            soft_table = soft_w = None
            if soft is not None:
                pairs = [(m, w) for m, w in zip(*soft[n]) if m.sum() >= a]
                soft_table, soft_w = measure_masks([m for m, _ in pairs]), [w for _, w in pairs]
            for c, mode, slot, q in slots:
                plan = plans[q]
                if soft is not None and plan["operation"] == "tertile_mean":
                    ans = answer_soft(soft_table, soft_w, plan["metric"])
                elif plan["operation"] in ("max", "min"):
                    ans = answer(ext_table, plan["metric"], plan["operation"], plan["fields"])
                else:
                    ans = answer(table, plan["metric"], plan["operation"], plan["fields"])
                preds[f"{n}.png"][f"{c}__{slot}"][mode] = {"final_answer": ans}
        records = score(preds, gts)
        med = {}
        for c in CATEGORY_METRIC:
            med[c] = []
            for slot in ("max", "tertile_mean"):
                for mode in ("clear", "open"):
                    errs = [r["pct_error"] for r in records if r["category"] == c and r["slot_type"] == slot
                            and r["mode"] == mode and r["pct_error"] is not None]
                    med[c].append(float(np.median(errs)) if errs else float("nan"))
        return records, {"median_ape": med, "mean_credit": float(np.mean([r["credit"] for r in records])),
                         "segmentation": {"precision": tp / max(npred, 1), "recall": tp / max(ngt, 1),
                                          "f1": 2 * tp / max(npred + ngt, 1)}}

    results = {}
    for regime in ("official", "debris_filtered"):
        results[regime] = {}
        for m, (masks, soft, extreme) in methods.items():
            records, results[regime][m] = evaluate(masks, soft, extreme, regime)
            (OUT / f"records_{regime}_{m}.json").write_text(json.dumps(records))

    (OUT / "results.json").write_text(json.dumps({"results": results, "published_morphoorgaagent": PUBLISHED,
                                                  "planner_correct": planner_ok, "n_questions": len(slots),
                                                  "n_images": len(names), "debris_min_area": DEBRIS_MIN_AREA}, indent=1))

    print(f"\nMorphoOrgaVQA: {len(names)} images x {len(slots)} questions. Planner picked the right metric + "
          f"operation for {planner_ok}/{len(slots)} questions.")
    for regime in ("official", "debris_filtered"):
        print(f"\n=== {regime} ground truth ===")
        print("Median APE (%)  [max clear | max open | tertile clear | tertile open]")
        for c in CATEGORY_METRIC:
            if regime == "official":
                print(f"  {c:10} MorphoOrgaAgent " + " | ".join(f"{v:6.2f}" for v in PUBLISHED[c]))
            for m in methods:
                print(f"  {c[:10] if regime != 'official' else '':10} {m:15} " +
                      " | ".join(f"{v:6.2f}" for v in results[regime][m]["median_ape"][c]))
        print("Mean credit and segmentation F1:")
        for m in methods:
            r = results[regime][m]
            s = r["segmentation"]
            print(f"  {m:12} credit {r['mean_credit']:.3f}   F1 {s['f1']:.3f} (P {s['precision']:.3f}, R {s['recall']:.3f})")


if __name__ == "__main__":
    main()
