"""Ask the VeriCell agent a question about an image from the command line.

    .venv/bin/python scripts/ask.py path/to/image.png "Which organoid is the largest?"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from vericell.agent import Agent  # noqa: E402


def main() -> None:
    image_path, question = Path(sys.argv[1]), sys.argv[2]
    cache = ROOT / "outputs" / "morphoorga" / "cache" if "morphoorga" in str(image_path) else None
    agent = Agent()
    for ev in agent.run(Image.open(image_path), question, name=image_path.stem if cache else None, cache=cache):
        print(f"[{ev['t']:6.1f}s] {ev['step']:8} {ev['title']}" + (f" — {ev['detail']}" if ev["detail"] else ""))
        if ev["step"] == "answer":
            r = ev["data"]["result"]
            print("\nreview queue (most answer-changing first):")
            for q in r["review"][:5]:
                print(f"   object {q['id']:4d}  P={q['p']:.2f}  impact {q['impact']:.1%}  {'; '.join(q['reasons'][:2])}")


if __name__ == "__main__":
    main()
