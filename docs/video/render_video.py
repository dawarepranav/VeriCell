"""Render the VeriCell demo video: compose.html frame by frame (30 fps) + voice-over + a soft pad -> MP4.

    python docs/video/render_video.py preview    # one frame per scene into build/preview
    python docs/video/render_video.py            # build/frames, build/mix.wav -> docs/video/VeriCell-demo.mp4

Needs: playwright (+ Chrome or `playwright install chromium`), ffmpeg, numpy; build/shots from capture.py,
build/vo + build/scenes_timed.json from the narration step.
"""

import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
FPS = 30
RATE = 48000


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/compose.html"


def frames(preview: bool) -> dict:
    srv, url = serve()
    out = BUILD / ("preview" if preview else "frames")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome")
        except Exception:
            b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        pg.goto(url, wait_until="networkidle")
        info = pg.evaluate("setup()")
        times = ([s["start"] + s["dur"] * f for s in info["scenes"] for f in (0.35, 0.75)] if preview
                 else [i / FPS for i in range(int(info["total"] * FPS))])
        for i, t in enumerate(times):
            pg.evaluate(f"render({t})")
            pg.screenshot(path=out / f"{i:05d}.jpg", type="jpeg", quality=93)
            if not preview and i % 300 == 0:
                print(f"frame {i}/{len(times)}", flush=True)
        b.close()
    srv.shutdown()
    return info


def audio(info: dict) -> Path:
    total = int(info["total"] * RATE) + RATE
    mix = np.zeros(total, dtype=np.float64)
    for s in info["scenes"]:
        wav = BUILD / "vo" / f"{s['id']}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", BUILD / "vo" / f"{s['id']}.aiff",
                        "-ar", str(RATE), "-ac", "1", wav], check=True)
        with wave.open(str(wav)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
        a = int(s["vo_at"] * RATE)
        mix[a:a + len(x)] += x[: total - a] * 0.95
    # a soft, slowly breathing pad (A minor colour) under the voice
    t = np.arange(total) / RATE
    pad = sum(np.sin(2 * np.pi * f * t + ph) for f, ph in ((110.0, 0), (164.81, 1.1), (220.0, 2.0), (261.63, .4)))
    pad *= (0.55 + 0.45 * np.sin(2 * np.pi * t / 9.0)) * 0.012
    env = np.clip(t / 2.5, 0, 1) * np.clip((info["total"] - t) / 3.0, 0, 1)
    mix += pad * env
    mix = np.clip(mix, -1, 1)
    path = BUILD / "mix.wav"
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    return path


def encode(mix: Path) -> Path:
    out = HERE / "VeriCell-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", BUILD / "frames" / "%05d.jpg",
                    "-i", mix, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out], check=True)
    return out


if __name__ == "__main__":
    preview = len(sys.argv) > 1 and sys.argv[1] == "preview"
    info = frames(preview)
    (BUILD / "timeline.json").write_text(json.dumps(info, indent=1))
    print(f"total {info['total']:.1f}s, {len(info['scenes'])} scenes")
    if not preview:
        print("encoded", encode(audio(info)))
