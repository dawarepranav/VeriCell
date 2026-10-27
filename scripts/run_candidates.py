"""Generate (and cache) every candidate source the verifier uses, for all LIVECell sample images.

Sources: Cellpose-SAM, SAM 3.1 text on zoomed tiles, and SAM 3.1 zoomed re-segmentation of each
Cellpose box (which gives every Cellpose cell a SAM confidence and a second mask).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import CACHE, cached  # noqa: E402
from vericell import api, data, proposers  # noqa: E402
from vericell.sam import SAM  # noqa: E402


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    sam = SAM(api.client())
    samples = list(data.livecell())
    cp = {name: cached(name, "cellpose", lambda image=image: proposers.cellpose(image)) for name, image, *_ in samples}

    def run(sample):
        name, image, *_ = sample
        t = cached(name, "sam_text_tiled", lambda: sam.segment_text_tiled(image, "cell"))
        z = cached(name, "hybrid_zoom", lambda: sam.segment_boxes_zoomed(image, [i.box for i in cp[name]]))
        print(f"{name}: cellpose {len(cp[name])}, sam tiled {len(t)}, zoom {len(z)}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, samples))


if __name__ == "__main__":
    main()
