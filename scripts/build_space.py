"""Assemble (and optionally publish) the Hugging Face Space for the public VeriCell demo.

The Space runs the web app in demo mode on a free CPU: the example images ship with their cached
segmentations, so no PyTorch or Cellpose is installed. Muse Spark and SAM 3.1 calls still go live
through the Meta Model API, using the Space's secrets (META_API_KEY, optional FIREWORKS_API_KEY).

    .venv/bin/python scripts/build_space.py                      # build dist/space/ only
    .venv/bin/python scripts/build_space.py --push USER/vericell # build and upload (needs `hf auth login`)
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "space"
EXAMPLES = ["organoid_mouse_test_image0017", "orgaextractor_21",
            "orgasegment_20210727_11562_ckufm1p2h4vzp0y7jh0fyg2yo", "organoid_orig_test_PDAC8"]

README = """---
title: VeriCell
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Microscopy AI that knows when it's wrong (Meta Muse Spark + SAM 3.1)
---

# VeriCell: microscopy AI that knows when it's wrong

Ask a question about a microscope image. VeriCell finds every organoid, checks each one, and answers
with an honest 90% range plus the few objects a human should double-check. Built for the Meta AI
Developer Hackathon 2026 on Muse Spark, SAM 3.1, Muse Voice Transcribe and Muse Spark Open (Fireworks).

This public demo uses four example images (uploads are off). Code, benchmarks and setup:
https://github.com/dawarepranav/VeriCell
"""

DOCKERFILE = """FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 VERICELL_DEMO=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "vericell.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

# Pinned to the versions the verifier pickles were trained with.
REQUIREMENTS = """numpy==2.4.6
scipy==1.17.1
pandas==3.1.0
scikit-learn==1.9.1
pillow>=10.0
opencv-python-headless>=4.8
pycocotools==2.0.11
openai>=1.40
httpx>=0.27
meta-sam-parser==0.1.0
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
"""


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "README.md").write_text(README)
    (OUT / "Dockerfile").write_text(DOCKERFILE)
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    shutil.copytree(ROOT / "vericell", OUT / "vericell", ignore=shutil.ignore_patterns("__pycache__"))
    (OUT / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "baseline.py", OUT / "scripts" / "baseline.py")  # cache (de)serialisation
    img_dir, cache_dir = OUT / "data" / "morphoorga" / "images", OUT / "outputs" / "morphoorga" / "cache"
    img_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    for n in EXAMPLES:
        shutil.copy(ROOT / "data" / "morphoorga" / "images" / f"{n}.png", img_dir)
        for f in (ROOT / "outputs" / "morphoorga" / "cache").glob(f"{n}.*"):
            shutil.copy(f, cache_dir)
    shutil.copytree(ROOT / "outputs" / "models", OUT / "outputs" / "models",
                    ignore=shutil.ignore_patterns("*.txt"))
    shutil.copytree(ROOT / "outputs" / "ui", OUT / "outputs" / "ui")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"built {OUT} ({size / 1e6:.1f} MB)")


def push(repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo, repo_type="space",
                      commit_message="Deploy VeriCell demo")
    print(f"pushed: https://huggingface.co/spaces/{repo}")
    print("Now add the secrets in the Space settings: META_API_KEY (and optionally FIREWORKS_API_KEY).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="USER/SPACE")
    args = ap.parse_args()
    build()
    if args.push:
        push(args.push)
