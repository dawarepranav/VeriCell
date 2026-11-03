"""Evaluate the crop judge: hosted Muse Spark vs Muse Spark Open, zero-shot vs visual few-shot.

Test cases are "touching parts joined into one object" candidates whose ground truth is clear:
one real organoid (IoU >= 0.5 with one GT object) or several (covers >= 2 GT objects by > 50%).
Few-shot reference crops come only from images whose index is odd; test cases only from even
images, so the judge never sees the test image among its examples.

    VERICELL_LLM is ignored here; both backends are evaluated explicitly.
    .venv/bin/python scripts/eval_judge.py [--n 80] [--shots 6] [--backends meta open]
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from vericell import proposers  # noqa: E402
from vericell.judge import crop_with_outline, judge  # noqa: E402
from vericell.metrics import iou_matrix  # noqa: E402
from vericell.sam import to_data_url  # noqa: E402

D, C, OUT = ROOT / "data" / "morphoorga", ROOT / "outputs" / "morphoorga" / "cache", ROOT / "outputs" / "judge_eval"


def cases():
    out = []
    for k, img_path in enumerate(sorted((D / "images").glob("*.png"))):
        n = img_path.stem
        merged = proposers.touching_clusters(load(C / f"{n}.cellpose.json"))
        if not merged:
            continue
        lab = np.load(D / "masks" / f"{n}.npy")
        gt = [lab == i for i in np.unique(lab) if i > 0]
        iou = iou_matrix([m.mask for m in merged], gt)
        for j, m in enumerate(merged):
            inside = [(m.mask & g).sum() / g.sum() for g in gt if (m.mask & g).any()]
            truth = "one_object" if iou[j].max() >= 0.5 else ("several_objects" if sum(f > 0.5 for f in inside) >= 2 else None)
            if truth:
                out.append({"image": n, "split": "example" if k % 2 else "test", "inst": m, "truth": truth})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="test cases (half one, half several)")
    ap.add_argument("--shots", type=int, default=6, help="reference crops for few-shot (half each class)")
    ap.add_argument("--backends", nargs="+", default=["meta", "open"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    all_cases = cases()
    test = [c for c in all_cases if c["split"] == "test"]
    pool = [c for c in all_cases if c["split"] == "example"]
    pick = []
    for t in ("one_object", "several_objects"):
        sub = [c for c in test if c["truth"] == t]
        pick += [sub[i] for i in rng.choice(len(sub), size=min(args.n // 2, len(sub)), replace=False)]
    examples = []
    images = {}
    img = lambda n: images.setdefault(n, Image.open(D / "images" / f"{n}.png").convert("RGB"))  # noqa: E731
    for t in ("one_object", "several_objects"):
        sub = [c for c in pool if c["truth"] == t]
        for i in rng.choice(len(sub), size=args.shots // 2, replace=False):
            c = sub[i]
            examples.append({"label": t, "image_url": to_data_url(crop_with_outline(img(c["image"]), c["inst"])),
                             "note": "Lobes, buds and internal cavities are normal in one organoid." if t == "one_object"
                             else "Separate bodies meet at a visible seam or gap."})
    print(f"test cases: {Counter(c['truth'] for c in pick)} from {len({c['image'] for c in pick})} images; "
          f"{len(examples)} reference crops from other images")

    results = {}
    for backend in args.backends:
        for shots in (0, len(examples)):
            ex = examples if shots else None

            def run(c):
                try:
                    return judge(None, img(c["image"]), c["inst"], "organoid", which=backend, examples=ex)["verdict"]
                except Exception as e:  # count failures as wrong, but report them
                    return f"error: {str(e)[:80]}"
            with ThreadPoolExecutor(6) as pool_ex:
                verdicts = list(pool_ex.map(run, pick))
            per = {t: np.mean([v == t for v, c in zip(verdicts, pick) if c["truth"] == t]) for t in ("one_object", "several_objects")}
            errors = sum(v.startswith("error") for v in verdicts)
            key = f"{backend}/{'few-shot' if shots else 'zero-shot'}"
            results[key] = {"balanced_accuracy": float(np.mean(list(per.values()))), "recall_one": float(per["one_object"]),
                            "recall_several": float(per["several_objects"]), "errors": errors, "n": len(pick)}
            print(f"{key:20} balanced acc {results[key]['balanced_accuracy']:.0%}  (one {per['one_object']:.0%}, "
                  f"several {per['several_objects']:.0%}){f'  errors {errors}' if errors else ''}", flush=True)
            if errors:
                print("   e.g.", next(v for v in verdicts if v.startswith("error")))
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
