"""Train the deployable verifiers the live agent uses (one for organoids, one for cells).

The benchmark numbers in RESULTS.md come from out-of-fold models; these are the final models fit on
all labelled images, saved to outputs/models/{organoid,cell}.pkl. Uses cached proposals, so it makes
no new API calls once the benchmark scripts have run.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import load  # noqa: E402
from vericell import api, data, pipeline  # noqa: E402
from vericell.features import FEATURES, build_candidates  # noqa: E402
from vericell.metrics import iou_matrix  # noqa: E402
from vericell.sam import SAM  # noqa: E402
from vericell.verifier import fit, matrix  # noqa: E402

MODELS = ROOT / "outputs" / "models"


def save(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    model = fit(X, y, groups)
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / f"{name}.pkl").write_bytes(pickle.dumps({"model": model, "features": FEATURES, "n_train": len(y),
                                                       "frac_correct": float(y.mean())}))
    print(f"{name}: trained on {len(y)} candidates from {len(np.unique(groups))} images ({y.mean():.0%} correct)")


def organoids() -> None:
    client = api.client()
    sam = SAM(client)
    D, cache = ROOT / "data" / "morphoorga", ROOT / "outputs" / "morphoorga" / "cache"
    Xs, ys, gs = [], [], []
    for k, img_path in enumerate(sorted((D / "images").glob("*.png"))):
        n = img_path.stem
        image = Image.open(img_path).convert("RGB")
        props = pipeline.propose(client, sam, image, name=n, cache=cache)
        cands = pipeline.candidates(image, props)
        merged = [i for i, c in enumerate(cands) if c.source == "merged"]
        pipeline.judge_candidates(client, image, cands, merged, props.scene["target_object"], name=n, cache=cache)
        lab = np.load(D / "masks" / f"{n}.npy")
        gt = [lab == i for i in np.unique(lab) if i > 0]
        Xs.append(matrix(cands))
        ys.append((iou_matrix([c.inst.mask for c in cands], gt).max(1) >= 0.5).astype(int))
        gs.append(np.full(len(cands), k))
    save("organoid", np.vstack(Xs), np.concatenate(ys), np.concatenate(gs))


def cells() -> None:
    cache = ROOT / "outputs" / "cache"
    Xs, ys, gs = [], [], []
    for k, (name, image, gt, _meta) in enumerate(data.livecell()):
        stem = name[:-4]
        cp, tx, zm = (load(cache / f"{stem}.{m}.json") for m in ("cellpose", "sam_text_tiled", "hybrid_zoom"))
        cands = build_candidates(image, cp, tx, zm)
        Xs.append(matrix(cands))
        ys.append((iou_matrix([c.inst.mask for c in cands], gt).max(1) >= 0.5).astype(int))
        gs.append(np.full(len(cands), k))
    save("cell", np.vstack(Xs), np.concatenate(ys), np.concatenate(gs))


if __name__ == "__main__":
    which = sys.argv[1:] or ["organoid", "cell"]
    if "organoid" in which:
        organoids()
    if "cell" in which:
        cells()
