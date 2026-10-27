"""VeriCell web UI: serves the page and the exported results (outputs/ui).

Run:
    .venv/bin/uvicorn vericell.web.app:app --port 8765
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "ui"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="VeriCell")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


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
