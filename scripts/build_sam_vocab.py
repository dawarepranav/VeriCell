"""Build VeriCell's SAM 3.1 vocabulary: which text prompts SAM actually segments well, per image type.

SAM 3.1 segments from a short phrase, and Muse Spark's first guess is usually the biology word
("organoid"), which SAM often doesn't see. This script measures instead of guessing:

  1. screen  every candidate phrase on a few TRAINING images per source dataset
  2. choose  the best single phrase and best pair (union) on the full training half
  3. test    the chosen pair on the HELD-OUT half against the phrases Muse Spark picked on its own

Split: even-index images of each source dataset train, odd-index images test. Every SAM call is
zoom-tiled exactly as in a live run and cached, so re-running costs nothing. Writes
vericell/sam_vocab.json (used by the agent) and outputs/vocab/report.json.

    .venv/bin/python scripts/build_sam_vocab.py
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import CACHE as LIVECELL_CACHE, load, save  # noqa: E402
from vericell import api, data, pipeline  # noqa: E402
from vericell.metrics import iou_matrix, match, scores  # noqa: E402
from vericell.sam import SAM, _dedupe  # noqa: E402

MO = ROOT / "data" / "morphoorga"
MO_CACHE = ROOT / "outputs" / "morphoorga" / "cache"
OUT = ROOT / "outputs" / "vocab"
VOCAB = ROOT / "vericell" / "sam_vocab.json"

CANDIDATES = {
    "organoid": ["organoid", "round organoid", "small round blobs", "round cell cluster", "translucent sphere",
                 "round translucent cyst", "circular cyst", "cell cluster", "dark cell clump", "spheroid",
                 "round blob", "bubble", "dark ring", "circle",
                 # 20 organoid-related phrases: the field's own words, and organoid + shape combinations
                 "intestinal organoid", "budding organoid", "cystic organoid", "tumor organoid", "tumoroid",
                 "enteroid", "colonoid", "epithelial cyst", "multicellular spheroid", "cell aggregate",
                 "small organoid", "large organoid", "round organoids", "organoids", "organoid cluster",
                 "lobed organoid", "hollow sphere", "round cyst", "cell sphere", "round organoid cluster"],
    "cell": ["cell", "cells", "round cell", "elongated cell", "small bright cell", "spindle-shaped cell",
             "cell body", "dark cell", "bright blob", "nucleus"],
}
SCREEN_PER_SOURCE = {"organoid": 4, "cell": 6}
TOP_FOR_PAIRS = 10  # pairs are searched among this many best single phrases (complements can rank low alone)


def slug(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")


def source(name: str) -> str:
    return name.split("_")[0]


def organoid_images() -> dict:
    """name -> (load image, load ground-truth masks); loaded on demand, ground truth is large."""
    def gt(name):
        labels = np.load(MO / "masks" / f"{name}.npy")
        return [labels == k for k in np.unique(labels) if k != 0]
    return {p.stem: (lambda p=p: Image.open(p).convert("RGB"), lambda n=p.stem: gt(n))
            for p in sorted((MO / "images").glob("*.png"))}


def cell_images() -> dict:
    out = {}
    for name, image, gt, _ in data.livecell():  # LIVECell sample is small: keep it in memory
        out[Path(name).stem] = (lambda image=image: image, lambda gt=gt: gt)
    return out


def split(names: list[str]) -> tuple[list[str], list[str]]:
    """Even index within each source dataset trains, odd index tests."""
    train, test = [], []
    by_src: dict[str, list[str]] = {}
    for n in sorted(names):
        by_src.setdefault(source(n), []).append(n)
    for group in by_src.values():
        for i, n in enumerate(group):
            (train if i % 2 == 0 else test).append(n)
    return train, test


def f1(pred, gt) -> dict:
    iou = iou_matrix([i.mask for i in pred], gt)
    _, pairs = match(iou)
    return scores(len(pred), len(gt), pairs, iou)


def main() -> None:
    sam = SAM(api.client())
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    report, vocab = {}, {}
    only = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. "cell": warm one kind's cache in parallel
    for kind, loader in (("organoid", organoid_images), ("cell", cell_images)):
        if only and kind != only:
            continue
        samples = loader()
        train, test = split(list(samples))

        def run(name: str, phrase: str):
            path = OUT / "cache" / f"{name}.{slug(phrase)}.json"
            if path.exists():
                return load(path)
            img = samples[name][0]()
            tiles, scale = pipeline.tiling(img)
            for attempt in range(3):  # SAM streams its result; a dropped connection is retried
                try:
                    inst = sam.segment_text_tiled(img, phrase, tiles=tiles, scale=scale)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            save(path, inst)
            return inst

        def sweep(names, phrases):
            jobs = [(n, ph) for n in names for ph in phrases]
            with ThreadPoolExecutor(3) as ex:
                outs = list(ex.map(lambda j: run(*j), jobs))
            return {j: o for j, o in zip(jobs, outs)}

        def mean_f1s(res, names, options):
            """Mean per-image F1 of each option (a list of phrases, unioned); ground truth loaded once per image."""
            per = np.zeros((len(names), len(options)))
            for a, n in enumerate(names):
                gt = samples[n][1]()
                for b, o in enumerate(options):
                    per[a, b] = f1(_dedupe(sum((res[(n, ph)] for ph in o), [])), gt)["f1"]
            return per.mean(0).tolist()

        # 1. screen every candidate on a few training images per source
        by_src: dict[str, list[str]] = {}
        for n in train:
            by_src.setdefault(source(n), []).append(n)
        screen = [n for g in by_src.values() for n in g[:SCREEN_PER_SOURCE[kind]]]
        res = sweep(screen, CANDIDATES[kind])
        screened = sorted(zip(mean_f1s(res, screen, [[ph] for ph in CANDIDATES[kind]]), CANDIDATES[kind]), reverse=True)
        print(f"\n[{kind}] screen on {len(screen)} training images:")
        for s, ph in screened:
            print(f"  {s:.3f}  {ph}")

        # 2. top phrases and their pairs on the full training half
        top = [ph for _, ph in screened[:TOP_FOR_PAIRS]]
        res |= sweep(train, top)
        options = [[ph] for ph in top] + [list(p) for p in combinations(top, 2)]
        ranked = sorted(zip(mean_f1s(res, train, options), options), key=lambda t: -t[0])
        best = ranked[0][1]
        print(f"[{kind}] best on {len(train)} training images: {best} (F1 {ranked[0][0]:.3f})")

        # 3. held-out half: the library vs what Muse Spark picked (organoids) or the plain word (cells)
        res |= sweep(test, best)
        rows = []
        for n in test:
            gt = samples[n][1]()
            lib = f1(_dedupe(sum((res[(n, ph)] for ph in best), [])), gt)
            if kind == "organoid" and (MO_CACHE / f"{n}.sam_text.json").exists():
                base_inst, base_phr = load(MO_CACHE / f"{n}.sam_text.json"), \
                    json.loads((MO_CACHE / f"{n}.scene.json").read_text())["sam_concepts"][:2]
            else:
                res |= sweep([n], ["cell"])
                base_inst, base_phr = res[(n, "cell")], ["cell"]
            base = f1(base_inst, gt)
            cp_path = (MO_CACHE / f"{n}.cellpose.json") if kind == "organoid" else (LIVECELL_CACHE / f"{n}.cellpose.json")
            n_cp = len(load(cp_path)) if cp_path.exists() else None
            lost = lambda k: n_cp is not None and k < 0.3 * max(n_cp, 1)  # noqa: E731  same rule as the agent
            rows.append({"image": n, "baseline_phrases": base_phr, "baseline_f1": base["f1"], "library_f1": lib["f1"],
                         "baseline_found": base["n_pred"], "library_found": lib["n_pred"], "gt": len(gt),
                         "cellpose_found": n_cp, "baseline_lost": lost(base["n_pred"]), "library_lost": lost(lib["n_pred"])})
        summary = {
            "train_images": len(train), "test_images": len(test), "phrases": best,
            "train_f1": ranked[0][0], "screen": [{"phrase": ph, "f1": s} for s, ph in screened],
            "pairs": [{"phrases": o, "f1": s} for s, o in ranked],
            "test_baseline_f1": float(np.mean([r["baseline_f1"] for r in rows])),
            "test_library_f1": float(np.mean([r["library_f1"] for r in rows])),
            "test_baseline_lost": int(sum(r["baseline_lost"] for r in rows)),
            "test_library_lost": int(sum(r["library_lost"] for r in rows)),
            "test_wins": int(sum(r["library_f1"] > r["baseline_f1"] + 1e-9 for r in rows)),
            "test_rows": rows,
        }
        print(f"[{kind}] held-out ({len(test)} images): F1 {summary['test_baseline_f1']:.3f} (baseline) → "
              f"{summary['test_library_f1']:.3f} (library); SAM 'lost' on {summary['test_baseline_lost']} → "
              f"{summary['test_library_lost']} images; library better on {summary['test_wins']}/{len(test)}")
        report[kind] = summary
        vocab[kind] = {"phrases": best, "train_f1": round(ranked[0][0], 3),
                       "test_f1": round(summary["test_library_f1"], 3),
                       "test_f1_before": round(summary["test_baseline_f1"], 3)}
    if only:  # a partial run only warms the cache; the full run writes the library
        print(f"warmed {only}")
        return
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    VOCAB.write_text(json.dumps(vocab, indent=1) + "\n")
    print(f"\nwrote {VOCAB.relative_to(ROOT)} and {(OUT / 'report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
