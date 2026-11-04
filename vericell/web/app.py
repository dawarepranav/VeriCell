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
ORGANOIDS = (ROOT / "data" / "morphoorga" / "images", ROOT / "outputs" / "morphoorga" / "cache", "organoid")
CELLS = (ROOT / "data" / "livecell" / "examples", ROOT / "outputs" / "livecell_examples" / "cache", "cell")
EXAMPLES = {  # demo images with cached proposals (fast); any upload runs live.  name -> (label, source)
    "organoid_mouse_test_image0017": ("Budding, mouse", ORGANOIDS),
    "orgaextractor_21": ("Dense colon", ORGANOIDS),
    "orgasegment_20210727_11562_ckufm1p2h4vzp0y7jh0fyg2yo": ("Intestinal", ORGANOIDS),
    "organoid_orig_test_PDAC8": ("Pancreatic cancer", ORGANOIDS),
    "organoid_orig_test_Lung2": ("Lung cancer", ORGANOIDS),
    "organoid_orig_test_ACC5": ("Adenoid cystic", ORGANOIDS),
    "organoid_mouse_test_b_image0009": ("Round, mouse", ORGANOIDS),
    "orgaextractor_30": ("Colon", ORGANOIDS),
    "A172_Phase_C7_1_00d08h00m_3": ("Glioblastoma", CELLS),
    "Huh7_Phase_A12_1_03d16h00m_4": ("Liver cancer", CELLS),
    "SHSY5Y_Phase_A10_1_01d08h00m_4": ("Neuroblastoma", CELLS),
    "MCF7_Phase_H4_1_01d08h00m_4": ("Breast cancer, dense", CELLS),
}


def example_files(name: str) -> tuple[Path, Path, str]:
    """(image path, cache dir, kind) of a demo example."""
    img_dir, cache_dir, kind = EXAMPLES[name][1]
    return img_dir / f"{name}.png", cache_dir, kind

app = FastAPI(title="VeriCell")
_agent = None

# Public demo mode: examples are pre-segmented; uploads are capped separately (their Cellpose-SAM runs
# on a GPU function) and a global cap on questions per hour keeps a public link from draining API credits.
DEMO = os.environ.get("VERICELL_DEMO") == "1"
MAX_ASKS_PER_HOUR = int(os.environ.get("VERICELL_MAX_ASKS_PER_HOUR", "40"))
_asks: deque = deque()


MAX_UPLOAD_BYTES = 15 * 2**20
MAX_SIDE = 1024 if DEMO else 2048  # the public demo runs uploads' Cellpose-SAM on CPU
MAX_UPLOADS_PER_HOUR = 15
_uploads: list[float] = []


def _upload_ok() -> bool:
    """Uploads run Cellpose-SAM on a GPU: a separate, smaller hourly cap in the public demo."""
    now = time.time()
    _uploads[:] = [t for t in _uploads if now - t < 3600]
    if len(_uploads) >= MAX_UPLOADS_PER_HOUR:
        return False
    _uploads.append(now)
    return True


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
    return [{"name": n, "label": label, "kind": example_files(n)[2]} for n, (label, _) in EXAMPLES.items()
            if example_files(n)[0].exists()]


@app.get("/api/example_image/{name}")
def example_image(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "unknown example")
    return FileResponse(example_files(name)[0])


@app.post("/api/ask")
async def ask(question: str = Form(...), example: str = Form(""), image: UploadFile | None = File(None),
              llm_backend: str = Form("meta")):
    """Streams the agent's steps as server-sent events; the last event carries the answer."""
    if not _rate_ok():
        raise HTTPException(429, "The public demo is busy (hourly question limit reached). Please try again later.")
    if example:
        if example not in EXAMPLES:
            raise HTTPException(404, "unknown example")
        path, cache, _ = example_files(example)
        img, name = Image.open(path), example
    elif image is not None:
        raw = await image.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Images up to {MAX_UPLOAD_BYTES // 2**20} MB, please.")
        if DEMO and not _upload_ok():
            raise HTTPException(429, "The public demo has analysed its hourly quota of uploads. Try an example, or again later.")
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception:
            raise HTTPException(400, "That file isn't an image VeriCell can read (PNG, JPEG or TIFF).")
        if max(img.size) > MAX_SIDE:  # keeps a run inside its time budget; measurements are in this image's pixels
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        name, cache = None, None
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


class Audit(BaseModel):
    session: str
    op: str  # "at_most" | "at_least"
    threshold: float


@app.post("/api/audit")
def audit(a: Audit):
    if a.session not in agent().sessions:
        raise HTTPException(404, "unknown session")
    try:
        return json.loads(_json(agent().audit(a.session, a.op, a.threshold)))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/session/{sid}/reimage")
def reimage(sid: str):
    if sid not in agent().sessions:
        raise HTTPException(404, "unknown session")
    return json.loads(_json(agent().reimage(sid)))


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
