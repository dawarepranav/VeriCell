"""Public VeriCell demo on Modal (CPU, demo mode: the four example images, hourly question cap).

Build the bundle, then deploy:
    .venv/bin/python scripts/build_space.py            # dist/space: app code + cached examples + models
    .venv/bin/modal secret create vericell-keys --from-dotenv .env   # META_API_KEY, FIREWORKS_API_KEY
    .venv/bin/modal deploy deploy/modal_app.py

The examples ship with cached segmentations, so no PyTorch or Cellpose runs here; Muse Spark and
SAM 3.1 calls go live through the Meta Model API with the keys from the Modal secret.
"""

from pathlib import Path

import modal

BUNDLE = Path(__file__).resolve().parent.parent / "dist" / "space"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(BUNDLE / "requirements.txt"))
    .env({"VERICELL_DEMO": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(BUNDLE), "/app")
)

app = modal.App("vericell")


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
