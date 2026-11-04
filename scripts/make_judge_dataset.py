"""Build a fine-tuning set for a crop judge from VeriCell's own ground-truth labels.

Each example = the judge's exact prompt (system + one outlined crop) and the right verdict as JSON:
  one_object       a candidate matching one ground-truth organoid (IoU >= 0.5)
  several_objects  a joined-lobes candidate covering >= 2 ground-truth organoids (> 50% each)
  not_an_object    a candidate touching no ground-truth organoid (IoU < 0.05 with all)
Ambiguous candidates (IoU 0.05-0.5) are left out. Only images with an ODD index are used, so the
80 test crops in scripts/eval_judge.py (even images) stay unseen.

    .venv/bin/python scripts/make_judge_dataset.py --per-class 700
Output: outputs/judge_ft/train.jsonl (OpenAI chat format, images as base64 data URIs)
"""

import argparse
import base64
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from vericell import proposers  # noqa: E402
from vericell.judge import SYSTEM, crop_with_outline  # noqa: E402
from vericell.metrics import iou_matrix  # noqa: E402

D, C, OUT = ROOT / "data" / "morphoorga", ROOT / "outputs" / "morphoorga" / "cache", ROOT / "outputs" / "judge_ft"
REASON = {"one_object": "One continuous body; lobes or cavities belong to the same organoid.",
          "several_objects": "Separate bodies meet at a visible seam or gap.",
          "not_an_object": "Debris, background or blur rather than an organoid."}


def jpeg_url(img: Image.Image, side: int = 384) -> str:
    img = img.copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def example(img: Image.Image, inst, label: str, target: str = "organoid") -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM.format(target=target)},
        {"role": "user", "content": [
            {"type": "text", "text": f"Is the red-outlined region one {target}, several, or not a {target}?"},
            {"type": "image_url", "image_url": {"url": jpeg_url(crop_with_outline(img, inst))}}]},
        {"role": "assistant", "content": json.dumps({"verdict": label, "confidence": 0.9, "reason": REASON[label]})},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--hard", action="store_true",
                    help="v2: focus on joined outlines (true one-organoid lobes vs glued neighbours)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pools = {k: [] for k in REASON}
    joined_one = []  # joined-lobe candidates that really are one organoid: the hard "one" cases
    for k, img_path in enumerate(sorted((D / "images").glob("*.png"))):
        if k % 2 == 0:
            continue  # even images are the judge test set
        n = img_path.stem
        lab = np.load(D / "masks" / f"{n}.npy")
        gt = [lab == i for i in np.unique(lab) if i > 0]
        cp = load(C / f"{n}.cellpose.json")
        merged = proposers.touching_clusters(cp)
        sam = load(C / f"{n}.sam_text.json")
        for inst, is_merged in [(i, False) for i in cp + sam] + [(i, True) for i in merged]:
            iou = iou_matrix([inst.mask], gt)[0] if gt else np.zeros(1)
            if iou.max() >= 0.5:
                pools["one_object"].append((n, inst))
                if is_merged:
                    joined_one.append((n, inst))
            elif iou.max() < 0.05:
                pools["not_an_object"].append((n, inst))
            elif is_merged:
                inside = [(inst.mask & g).sum() / g.sum() for g in gt if (inst.mask & g).any()]
                if sum(f > 0.5 for f in inside) >= 2:
                    pools["several_objects"].append((n, inst))
        # Real "glued" errors are rare, so add the same mistake made on purpose: two neighbouring
        # Cellpose objects that each match a DIFFERENT ground-truth organoid, joined into one outline.
        good = [(i, int(np.argmax(r))) for i, r in zip(cp, iou_matrix([c.mask for c in cp], gt)) if r.max() >= 0.5] if gt else []
        for a in range(len(good)):
            for b in range(a + 1, len(good)):
                (ia, ga), (ib, gb) = good[a], good[b]
                gap = max(ia.box[0] - ib.box[2], ib.box[0] - ia.box[2], ia.box[1] - ib.box[3], ib.box[1] - ia.box[3])
                if ga != gb and gap <= 6:
                    joined = proposers.touching_clusters([ia, ib], gap=8)
                    if joined:
                        pools["several_objects"].append((n, joined[0]))
    print("available:", {k: len(v) for k, v in pools.items()}, "joined-lobe one:", len(joined_one))
    OUT.mkdir(parents=True, exist_ok=True)
    images, rows = {}, []

    def add(items, label, repeat=1):
        for n, inst in items:
            img = images.setdefault(n, Image.open(D / "images" / f"{n}.png").convert("RGB"))
            rows.extend([example(img, inst, label)] * repeat)

    def sample(pool, k):
        return [pool[i] for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]

    if args.hard:
        # v1 learned "a joined outline means several": its joined outlines were almost all glued pairs.
        # Here every true joined-lobe organoid is shown (x3) against the glued pairs (x2).
        add(joined_one, "one_object", repeat=3)
        add(sample([x for x in pools["one_object"] if not any(x[1] is j[1] for j in joined_one)], 300), "one_object")
        add(pools["several_objects"], "several_objects", repeat=2)
        add(sample(pools["not_an_object"], 300), "not_an_object")
    else:
        for label, pool in pools.items():
            add(sample(pool, args.per_class), label)
    rng.shuffle(rows)
    name = "train_v2.jsonl" if args.hard else "train.jsonl"
    with open(OUT / name, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    sizes = Counter(json.loads(r["messages"][2]["content"])["verdict"] for r in rows)
    print(f"wrote {len(rows)} examples {dict(sizes)} -> {OUT / name} ({(OUT / name).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
