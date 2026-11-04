"""VeriCell web app: the live agent (ask by text or voice, watch it work, review its doubts) and the
benchmark explorer (outputs/ui).

Run:
    .venv/bin/uvicorn vericell.web.app:app --port 8765
"""

import io
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "ui"
STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = {  # demo images with cached proposals (instant); any upload runs live
    "organoid_mouse_test_image0017": "Mouse intestinal organoids (budding)",
    "orgaextractor_21": "Colon organoids, dense",
    "orgasegment_20210727_11562_ckufm1p2h4vzp0y7jh0fyg2yo": "Intestinal organoids",
    "organoid_orig_test_PDAC8": "Pancreatic cancer organoids",
}
EXAMPLE_DIR = ROOT / "data" / "morphoorga" / "images"
EXAMPLE_CACHE = ROOT / "outputs" / "morphoorga" / "cache"

app = FastAPI(title="VeriCell")
_agent = None

# Public demo mode (e.g. a Hugging Face Space): uploads off (no Cellpose on a free CPU; examples are
# pre-segmented) and a global cap on questions per hour so a public link can't drain the API credits.
DEMO = os.environ.get("VERICELL_DEMO") == "1"
MAX_ASKS_PER_HOUR = int(os.environ.get("VERICELL_MAX_ASKS_PER_HOUR", "40"))
_asks: deque = deque()


def _rate_ok() -> bool:
    now = time.time()
    while _asks and now - _asks[0] > 3600:
        _asks.popleft()
    if DEMO and len(_asks) >= MAX_ASKS_PER_HOUR:
        return False
    _asks.append(now)
    return True


def agent():
    global _agent
    if _agent is None:
        from vericell.agent import Agent
        _agent = Agent()
    return _agent


def _json(obj):
    return json.dumps(obj, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


@app.get("/")
def landing():
    return FileResponse(STATIC / "landing.html")


@app.get("/app")
def lab():
    return FileResponse(STATIC / "index.html")


# ---------------------------------------------------------------- live agent
@app.get("/api/config")
def config():
    from vericell.api import load_key
    load_key()  # also loads .env, so a local FIREWORKS_API_KEY is seen
    return {"demo": DEMO, "open_weights": bool(os.environ.get("FIREWORKS_API_KEY"))}


@app.get("/api/examples")
def examples():
    return [{"name": n, "label": label} for n, label in EXAMPLES.items() if (EXAMPLE_DIR / f"{n}.png").exists()]


@app.get("/api/example_image/{name}")
def example_image(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "unknown example")
    return FileResponse(EXAMPLE_DIR / f"{name}.png")


@app.post("/api/ask")
async def ask(question: str = Form(...), example: str = Form(""), image: UploadFile | None = File(None),
              llm_backend: str = Form("meta")):
    """Streams the agent's steps as server-sent events; the last event carries the answer."""
    if not _rate_ok():
        raise HTTPException(429, "The public demo is busy (hourly question limit reached). Please try again later.")
    if example:
        if example not in EXAMPLES:
            raise HTTPException(404, "unknown example")
        img, name, cache = Image.open(EXAMPLE_DIR / f"{example}.png"), example, EXAMPLE_CACHE
    elif image is not None:
        if DEMO:
            raise HTTPException(403, "Uploads are off in the public demo; run VeriCell locally to analyse your own images.")
        img, name, cache = Image.open(io.BytesIO(await image.read())), None, None
    else:
        raise HTTPException(400, "send an image or pick an example")
    a = agent()

    def stream():
        try:
            for ev in a.run(img, question, name=name, cache=cache, llm_backend=llm_backend):
                yield f"data: {_json(ev)}\n\n"
        except Exception as exc:  # surface failures in the trace instead of a dead stream
            yield f"data: {_json({'step': 'error', 'title': 'Something went wrong', 'detail': str(exc), 'data': {}})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/session/{sid}/image.png")
def session_image(sid: str):
    sess = agent().sessions.get(sid)
    if sess is None:
        raise HTTPException(404, "unknown session")
    buf = io.BytesIO()
    sess.image.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


class Review(BaseModel):
    session: str
    object_id: int
    real: bool | None  # None undoes an earlier check


@app.post("/api/review")
def review(r: Review):
    if r.session not in agent().sessions:
        raise HTTPException(404, "unknown session")
    return json.loads(_json(agent().review(r.session, r.object_id, r.real)))


@app.get("/api/session/{sid}/evidence")
def evidence(sid: str):
    if sid not in agent().sessions:
        raise HTTPException(404, "unknown session")
    body = _json(agent().evidence(sid))
    return Response(body, media_type="application/json",
                    headers={"content-disposition": f'attachment; filename="vericell-evidence-{sid}.json"'})


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    from vericell.voice import transcribe as run
    return {"text": run(await audio.read())}


# ---------------------------------------------------------------- benchmark explorer
@app.get("/api/summary")
def summary():
    path = DATA / "summary.json"
    if not path.exists():
        raise HTTPException(404, "No results yet: run scripts/build_results.py")
    return json.loads(path.read_text())


@app.get("/api/image/{stem}")
def image(stem: str):
    path = DATA / "cells" / f"{Path(stem).name}.json"
    if not path.exists():
        raise HTTPException(404, f"Unknown image {stem}")
    return json.loads(path.read_text())


app.mount("/img", StaticFiles(directory=DATA / "img", check_dir=False), name="img")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
