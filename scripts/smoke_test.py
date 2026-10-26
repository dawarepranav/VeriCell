"""Smoke test: confirm the Meta Model API key works for every model VeriCell needs.

Checks, in order:
  1. the key is accepted and the required model IDs are listed
  2. Muse Spark answers a text prompt
  3. Muse Spark reads an image (synthetic "cells")
  4. SAM 3.1 segments that image from a concept prompt, with per-object confidence
  5. SAM 3.1 segments one object from a box prompt
  6. Muse Voice Transcribe transcribes a short spoken question (macOS `say` makes the audio)

Usage:
    export META_API_KEY=...   # or MODEL_API_KEY, or put it in .env
    .venv/bin/python scripts/smoke_test.py
"""

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image

BASE_URL = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
TRANSCRIBE = "muse-voice-transcribe-1.0"
ROOT = Path(__file__).resolve().parent.parent


def load_key() -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() and not line.lstrip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("No API key found. Set META_API_KEY (or MODEL_API_KEY) in your shell or in .env.")
    return key


def synthetic_cells(n: int = 12, size: int = 256, seed: int = 0):
    """Dark background with n bright disks. Returns (PIL image, list of (cx, cy, r))."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = rng.normal(20, 4, (size, size))
    cells = []
    while len(cells) < n:
        cx, cy = rng.uniform(20, size - 20, 2)
        r = rng.uniform(8, 14)
        if all((cx - x) ** 2 + (cy - y) ** 2 > (r + rr + 4) ** 2 for x, y, rr in cells):
            cells.append((cx, cy, r))
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < r**2] = rng.uniform(150, 220)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L"), cells


def data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sam_request(client: OpenAI, text: str, image: Image.Image) -> str:
    """SAM 3.1 streams its result as box/mask tokens; collect the full text."""
    stream = client.responses.create(
        model=SAM,
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": data_url(image)},
            ],
        }],
        stream=True,
        metadata={"include_confidence": "true"},
    )
    return "".join(ev.delta for ev in stream if ev.type == "response.output_text.delta")


def summarize_sam(raw: str) -> str:
    boxes = re.findall(r"<\|box;[^|]*\|>", raw)
    conf = [float(c) for c in re.findall(r"\bc=([0-9.]+)", raw)]
    conf_txt = f", confidence {min(conf):.2f}-{max(conf):.2f}" if conf else ", no confidence values found"
    return f"{len(boxes)} object(s){conf_txt}"


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key=load_key())
    image, cells = synthetic_cells()
    results = []

    def check(name, fn):
        start = time.time()
        try:
            detail = fn()
            results.append(True)
            print(f"PASS  {name} ({time.time() - start:.1f}s): {detail}")
        except Exception as exc:  # report every failure, keep going
            results.append(False)
            print(f"FAIL  {name} ({time.time() - start:.1f}s): {type(exc).__name__}: {exc}")

    def list_models():
        ids = {m.id for m in client.models.list()}
        missing = {SPARK, SAM, TRANSCRIBE} - ids
        if missing:
            raise RuntimeError(f"key works but these models are not listed: {sorted(missing)}; got {sorted(ids)}")
        return f"{len(ids)} models listed, all required ones present"

    def spark_text():
        r = client.responses.create(model=SPARK, input="Reply with exactly the word OK.")
        return repr(r.output_text.strip()[:80])

    def spark_vision():
        r = client.responses.create(
            model=SPARK,
            input=[{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "How many bright round objects are in this image? Reply with a number only."},
                    {"type": "input_image", "image_url": data_url(image)},
                ],
            }],
        )
        return f"answered {r.output_text.strip()[:40]!r} (true count {len(cells)})"

    def sam_concept():
        raw = sam_request(client, "circle", image)
        return f"{summarize_sam(raw)} (true count {len(cells)})"

    def sam_box():
        cx, cy, r = cells[0]
        w, h = image.size
        x1, y1, x2, y2 = (int(max(0, cx - r - 3)), int(max(0, cy - r - 3)),
                          int(min(w - 1, cx + r + 3)), int(min(h - 1, cy + r + 3)))
        raw = sam_request(client, f"<0f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>", image)
        return f"{summarize_sam(raw)} (expected 1)"

    def transcribe():
        if not shutil.which("say"):
            raise RuntimeError("skipped: macOS `say` not available to generate test audio")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "question.wav"
            subprocess.run(["say", "-o", str(wav), "--data-format=LEI16@16000",
                            "How many of these cells are dividing?"], check=True)
            resp = httpx.post(
                f"{BASE_URL}/asr/transcribe",
                headers={"Authorization": f"Bearer {client.api_key}"},
                # The endpoint wants two multipart parts: a JSON "request" and the "audio" bytes.
                files={
                    "request": (None, json.dumps({"model": TRANSCRIBE, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}), "application/json"),
                    "audio": ("question.wav", wav.read_bytes(), "audio/wav"),
                },
                timeout=60,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.text[:200]

    check("API key + model list", list_models)
    check(f"{SPARK} text", spark_text)
    check(f"{SPARK} vision", spark_vision)
    check(f"{SAM} concept prompt", sam_concept)
    check(f"{SAM} box prompt", sam_box)
    check(f"{TRANSCRIBE} file upload", transcribe)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
