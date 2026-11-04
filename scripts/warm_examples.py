"""Pre-compute the demo examples: one real run each with caching on, so the public demo answers fast.

Caches Cellpose-SAM (every pyramid scale), SAM 3.1 with the measured vocabulary, zoomed second
opinions, and the scene description next to each example (see vericell/web/app.py EXAMPLES).
Muse Spark's planning, judging and agent decisions still run live in the demo.

    .venv/bin/python scripts/warm_examples.py            # all examples
    .venv/bin/python scripts/warm_examples.py NAME ...   # some
"""

import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vericell.agent import Agent  # noqa: E402
from vericell.web.app import EXAMPLES, example_files  # noqa: E402


def main(names: list[str]) -> None:
    agent = Agent()
    for name in names or list(EXAMPLES):
        path, cache, kind = example_files(name)
        cache.mkdir(parents=True, exist_ok=True)
        t0, last = time.time(), None
        for ev in agent.run(Image.open(path), f"Which {kind} is the largest, and where is it?", name=name, cache=cache):
            last = ev
        print(f"{name}: {time.time() - t0:.0f}s · {last['title'][:90]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
