"""Candidate generation for one image: every proposer the verifier chooses between.

Shared by the live agent and the training script. Results can be cached per image name so
re-running on benchmark images costs no API calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image

from . import proposers
from .features import Candidate, add_judgement, build_candidates
from .judge import judge
from .planner import describe_image
from .sam import SAM, Instance, _dedupe
from .verifier import matrix


def tiling(image: Image.Image) -> tuple[int, float]:
    """Tiles per side and upscale so each tile reaches SAM's ~1008 px working size."""
    side = max(image.size)
    tiles = int(np.clip(round(side / 700), 1, 4))
    return tiles, float(np.clip(1008 / (side / tiles), 1.0, 3.0))


def _cached(path: Path | None, fn):
    from scripts.baseline import load, save  # RLE (de)serialisation lives with the baseline script

    if path is not None and path.exists():
        return load(path)
    inst = fn()
    if path is not None:
        save(path, inst)
    return inst


@dataclass
class Proposals:
    scene: dict
    cellpose: list[Instance]
    coarse: list[Instance]
    merged: list[Instance]
    sam: list[Instance]
    zoom: list[Instance]
    timings: dict = field(default_factory=dict)


def propose(client: OpenAI, sam: SAM, image: Image.Image, name: str | None = None,
            cache: Path | None = None, pyramid: bool = True) -> Proposals:
    """Scene description, Cellpose-SAM (+ pyramid), SAM 3.1 text on zoomed tiles, zoomed box re-segmentation."""
    import time

    def path(suffix):
        return cache / f"{name}.{suffix}" if cache is not None and name else None

    t = {}
    t0 = time.time()
    sp = path("scene.json")
    if sp is not None and sp.exists():
        scene = json.loads(sp.read_text())
    else:
        scene = describe_image(client, image)
        if sp is not None:
            sp.write_text(json.dumps(scene))
    t["scene"] = time.time() - t0

    t0 = time.time()
    cp = _cached(path("cellpose.json"), lambda: proposers.cellpose(image))
    coarse = []
    if pyramid:
        for s in proposers.pyramid_scales(image)[1:]:
            coarse += _cached(path(f"cellpose_s{s:.3f}.json"), lambda s=s: proposers.cellpose_scaled(image, s))
    merged = proposers.touching_clusters(cp) if pyramid else []
    t["cellpose"] = time.time() - t0

    t0 = time.time()
    tiles, scale = tiling(image)

    def text():
        found = []
        for concept in scene["sam_concepts"][:2]:
            found += sam.segment_text_tiled(image, concept, tiles=tiles, scale=scale)
        return _dedupe(found)

    tx = _cached(path("sam_text.json"), text)
    zm = _cached(path("zoom.json"), lambda: sam.segment_boxes_zoomed(image, [i.box for i in cp]))
    t["sam"] = time.time() - t0
    return Proposals(scene, cp, coarse, merged, tx, zm, t)


def candidates(image: Image.Image, props: Proposals) -> list[Candidate]:
    extra = {"coarse": props.coarse, "merged": props.merged} if (props.coarse or props.merged) else None
    return build_candidates(image, props.cellpose, props.sam, props.zoom, extra=extra)


def judge_candidates(client: OpenAI, image: Image.Image, cands: list[Candidate], which: list[int],
                     target: str, name: str | None = None, cache: Path | None = None) -> list[int]:
    """Ask the Muse Spark crop judge about candidates `which` (cached by box). Returns the indices judged."""
    from concurrent.futures import ThreadPoolExecutor

    jp = cache / f"{name}.judge.json" if cache is not None and name else None
    store = json.loads(jp.read_text()) if jp is not None and jp.exists() else {}
    todo = [i for i in which if str(list(cands[i].inst.box)) not in store]
    with ThreadPoolExecutor(8) as ex:
        for i, v in zip(todo, ex.map(lambda i: judge(client, image, cands[i].inst, target), todo)):
            store[str(list(cands[i].inst.box))] = v
    if jp is not None and todo:
        jp.write_text(json.dumps(store))
    for i in which:
        add_judgement(cands[i], store[str(list(cands[i].inst.box))])
    return which


def features(cands: list[Candidate]) -> np.ndarray:
    return matrix(cands)
