"""Fragility study: how often could one uncertain object change a "most extreme organoid" conclusion,
does VeriCell's warning predict wrong answers, and how many wrong conclusions does checking only the
objects VeriCell names fix?

MorphoOrgaVQA, 68 images x 4 extreme questions (largest area, longest perimeter, roundest, roughest).
Uses the benchmark's cached segmentations and crop-judge verdicts and OUT-OF-FOLD verifier
probabilities (no image is scored by a verifier trained on it); no API calls. The "human check" is
simulated with ground truth: an object the human is shown is resolved as real if it matches a
ground-truth organoid (IoU >= 0.5), else not real.

    .venv/bin/python scripts/fragility_study.py      # writes outputs/fragility/study.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from vericell import answers, proposers  # noqa: E402
from vericell.features import add_judgement, build_candidates  # noqa: E402
from vericell.measure import measure_masks  # noqa: E402
from vericell.metrics import iou_matrix  # noqa: E402
from vericell.verifier import matrix, out_of_fold  # noqa: E402

DATA = ROOT / "data" / "morphoorga"
CACHE = ROOT / "outputs" / "morphoorga" / "cache"
OUT = ROOT / "outputs" / "fragility"
QUESTIONS = {"largest area": ("Area_outer", "max"), "longest perimeter": ("Perimeter_outer", "max"),
             "roundest": ("roundness", "max"), "roughest edge": ("perimeter_diff", "max")}
CREDIT = 0.15      # the benchmark's full-credit error threshold
NEAR_TIE = 0.05    # same rules as the agent's verification (vericell/agent.py)


def verdict(st: dict) -> tuple[bool, list[int]]:
    """VeriCell's 'could the conclusion change?' and the objects it would show a human."""
    watch = [c["id"] for c in st["challengers"][:2]]
    w, nxt = st["winner"], st["successor"]
    if w["p"] < 0.9 and nxt:
        watch.append(w["id"])
    if nxt and abs(nxt["gap"]) < NEAR_TIE:
        watch.append(nxt["id"])
    return bool(watch), list(dict.fromkeys(watch))


def main() -> None:
    names = sorted(p.stem for p in (DATA / "images").glob("*.png"))
    cands, gt_tables, truth = {}, {}, {}
    for n in names:
        img = Image.open(DATA / "images" / f"{n}.png").convert("RGB")
        cp = load(CACHE / f"{n}.cellpose.json")
        coarse = [i for s in proposers.pyramid_scales(img)[1:] for i in load(CACHE / f"{n}.cellpose_s{s:.3f}.json")]
        c = build_candidates(img, cp, load(CACHE / f"{n}.sam_text.json"), load(CACHE / f"{n}.zoom.json"),
                             extra={"coarse": coarse, "merged": proposers.touching_clusters(cp)})
        judged = json.loads((CACHE / f"{n}.judge.json").read_text()) if (CACHE / f"{n}.judge.json").exists() else {}
        for x in c:
            if x.source == "merged" and str(list(x.inst.box)) in judged:
                add_judgement(x, judged[str(list(x.inst.box))])
        lab = np.load(DATA / "masks" / f"{n}.npy")
        gt = [lab == k for k in np.unique(lab) if k > 0]
        truth[n] = iou_matrix([x.inst.mask for x in c], gt).max(1) >= 0.5 if c else np.zeros(0, bool)
        gt_tables[n] = measure_masks(gt)
        cands[n] = c
        print(f"{n}: {len(c)} candidates", flush=True)

    X = np.vstack([matrix(cands[n]) for n in names])
    y = np.concatenate([truth[n].astype(int) for n in names])
    groups = np.concatenate([[k] * len(cands[n]) for k, n in enumerate(names)])
    p_all = out_of_fold(X, y, groups)

    rows, off = [], 0
    for n in names:
        k = len(cands[n])
        p = p_all[off:off + k]
        off += k
        table = [(measure_masks([x.inst.mask]) or [None])[0] for x in cands[n]]
        idx = answers.pool(cands[n], p, table)
        for q, (metric, op) in QUESTIONS.items():
            plan = {"metric": metric, "operation": op, "fields": [metric]}
            value, winner = answers.point(plan, table, p, idx)
            if not value or not gt_tables[n]:
                continue
            true = max(r[metric] for r in gt_tables[n])
            err = abs(value[metric] - true) / max(abs(true), 1e-9)
            flagged, watch = verdict(answers.stability(plan, table, p, idx, winner))
            # a human checks only the objects VeriCell names (ground truth stands in for the human)
            q2 = p.copy()
            for i in watch:
                q2[i] = 1.0 if truth[n][i] else 0.0
            v2, _ = answers.point(plan, table, q2, idx)
            err2 = abs(v2[metric] - true) / max(abs(true), 1e-9)
            rows.append({"image": n, "question": q, "error": err, "flagged": flagged, "n_checked": len(watch),
                         "error_after_check": err2, "winner_p": float(p[winner])})

    def summarise(rs):
        wrong = [r for r in rs if r["error"] > CREDIT]
        flagged = [r for r in rs if r["flagged"]]
        held = [r for r in rs if not r["flagged"]]
        fixed = [r for r in wrong if r["flagged"] and r["error_after_check"] <= CREDIT]
        broken = [r for r in rs if r["error"] <= CREDIT and r["error_after_check"] > CREDIT]
        rate = lambda xs: sum(r["error"] > CREDIT for r in xs) / len(xs) if xs else None  # noqa: E731
        return {
            "conclusions": len(rs), "wrong": len(wrong), "flagged": len(flagged),
            "wrong_rate_when_flagged": rate(flagged), "wrong_rate_when_holds": rate(held),
            "wrong_caught_by_flag": sum(r["flagged"] for r in wrong) / len(wrong) if wrong else None,
            "median_error_flagged": float(np.median([r["error"] for r in flagged])) if flagged else None,
            "median_error_holds": float(np.median([r["error"] for r in held])) if held else None,
            "objects_checked_when_flagged": float(np.mean([r["n_checked"] for r in flagged])) if flagged else 0,
            "wrong_fixed_by_checking": len(fixed), "right_made_wrong_by_checking": len(broken),
            "wrong_after_checking": sum(r["error_after_check"] > CREDIT for r in rs),
        }

    study = {"all": summarise(rows), "by_question": {q: summarise([r for r in rows if r["question"] == q]) for q in QUESTIONS},
             "rows": rows, "credit_threshold": CREDIT}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "study.json").write_text(json.dumps(study, indent=1))
    a = study["all"]
    print(f"\n{a['conclusions']} conclusions, {a['wrong']} wrong (error > {CREDIT:.0%}); flagged {a['flagged']}")
    print(f"wrong when flagged: {a['wrong_rate_when_flagged']:.0%} · wrong when VeriCell says it holds: {a['wrong_rate_when_holds']:.0%}")
    print(f"wrong conclusions the flag caught: {a['wrong_caught_by_flag']:.0%}")
    print(f"checking ~{a['objects_checked_when_flagged']:.1f} objects per flagged conclusion fixes {a['wrong_fixed_by_checking']}"
          f" of {a['wrong']} wrong conclusions (and breaks {a['right_made_wrong_by_checking']}); wrong after: {a['wrong_after_checking']}")
    for q, s in study["by_question"].items():
        print(f"  {q:18} wrong {s['wrong']:3}/{s['conclusions']}  flagged-wrong {s['wrong_rate_when_flagged'] or 0:.0%}  "
              f"held-wrong {s['wrong_rate_when_holds'] or 0:.0%}  fixed {s['wrong_fixed_by_checking']}")


if __name__ == "__main__":
    main()
