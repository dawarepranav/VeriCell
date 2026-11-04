"""Public VeriCell demo on Modal: a small web app (demo mode, hourly caps) plus a separate Cellpose-SAM
function that wakes only when someone uploads their own image.

Build the bundle, then deploy:
    .venv/bin/python scripts/build_space.py            # dist/space: app code + cached examples + models
    .venv/bin/modal secret create vericell-keys --from-dotenv .env   # META_API_KEY, FIREWORKS_API_KEY
    .venv/bin/modal deploy deploy/modal_app.py

The examples ship with cached segmentations, so the web container has no PyTorch; uploads send their
Cellpose-SAM passes to `cellpose_labels` (8 CPU cores, scales to zero; set CELLPOSE_GPU=T4 at deploy
time to use a GPU instead, which needs a payment method on the Modal account). Muse Spark and SAM 3.1 calls go live
through the Meta Model API with the keys from the Modal secret.
"""

import os
from pathlib import Path

import modal

BUNDLE = Path(__file__).resolve().parent.parent / "dist" / "space"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(BUNDLE / "requirements.txt"))
    .pip_install("modal")
    .env({"VERICELL_DEMO": "1", "PYTHONUNBUFFERED": "1", "VERICELL_CELLPOSE": "modal"})
    .add_local_dir(str(BUNDLE), "/app")
)

app = modal.App("vericell")


def _fetch_weights():
    from cellpose import models

    models.CellposeModel(gpu=False)  # downloads the Cellpose-SAM weights into the image


CELLPOSE_GPU = os.environ.get("CELLPOSE_GPU")  # e.g. "T4"; None = CPU

cellpose_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.14.1", "cellpose==4.2.1.1", "numpy==2.4.6", "opencv-python-headless>=4.8")
    .run_function(_fetch_weights)
)


@app.function(image=cellpose_image, gpu=CELLPOSE_GPU, cpu=8.0, memory=8192, max_containers=2,
              scaledown_window=120, timeout=600)
def cellpose_labels(arr_npy: bytes, kw: dict) -> bytes:
    """Cellpose-SAM label image for one array (as .npy bytes)."""
    import io

    import numpy as np
    from cellpose import models

    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = models.CellposeModel(gpu=CELLPOSE_GPU is not None)
    labels, _f, _s = _MODEL.eval(np.load(io.BytesIO(arr_npy)), **kw)
    buf = io.BytesIO()
    np.save(buf, labels.astype(np.int32))
    return buf.getvalue()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("vericell-keys")],
    # Review clicks must reach the container that holds the session, so keep exactly one.
    max_containers=1,
    scaledown_window=900,
    timeout=600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    from vericell.web.app import app as fastapi_app

    return fastapi_app
