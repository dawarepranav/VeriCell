"""Serve and evaluate the fine-tuned Muse Glimmer crop judge on Fireworks.

The fine-tuned adapter is served by one "live merge" deployment on a single H200 (the multi-LoRA
GB300 shape has no quota on this account). It is compared with hosted Muse Spark and Muse Spark Open
(scripts/eval_judge.py) on the same 80 test crops, images the fine-tune never saw, prompted exactly as
in training.

    .venv/bin/python scripts/glimmer_judge.py up       # create the deployment, load the adapter
    .venv/bin/python scripts/glimmer_judge.py status
    .venv/bin/python scripts/glimmer_judge.py eval     # base vs fine-tuned on the 80 test crops
    .venv/bin/python scripts/glimmer_judge.py down     # delete the deployment (stops all GPU cost)

Set GLIMMER_JUDGE=<model id> to serve another fine-tune (default: vericell-judge-glimmer).
"""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_judge import D, cases  # noqa: E402
from scripts.make_judge_dataset import example  # noqa: E402
from vericell.api import load_key  # noqa: E402
from vericell.llm import FIREWORKS_URL, _first_json  # noqa: E402

ACCOUNT = "accounts/pranavdaware1-1-xupq"
API = f"https://api.fireworks.ai/v1/{ACCOUNT}"
BASE = "accounts/fireworks/models/muse-glimmer-30b"
TUNED = f"{ACCOUNT}/models/" + os.environ.get("GLIMMER_JUDGE", "vericell-judge-glimmer")
DEPLOYMENT = os.environ.get("GLIMMER_DEPLOYMENT", "vericell-judge-eval")
SHAPE = "accounts/fireworks/deploymentShapes/dkuwotf7"  # 1x NVIDIA H200 141GB, BF16, matched for the adapter
OUT = ROOT / "outputs" / "judge_eval"


def headers() -> dict:
    load_key()
    return {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}


def up() -> None:
    r = httpx.post(f"{API}/deployments", headers=headers(), params={"deploymentId": DEPLOYMENT}, timeout=120, json={
        "baseModel": TUNED, "deploymentShape": SHAPE,
        "minReplicaCount": 1, "maxReplicaCount": 1, "displayName": "VeriCell judge evaluation"})
    print("create deployment:", r.status_code, r.text[:300])


def load_adapter() -> None:
    r = httpx.post(f"{API}/deployedModels", headers=headers(), timeout=120, json={
        "model": TUNED, "deployment": f"{ACCOUNT}/deployments/{DEPLOYMENT}", "default": False, "serverless": False})
    print("load adapter:", r.status_code, r.text[:300])


def status() -> None:
    r = httpx.get(f"{API}/deployments/{DEPLOYMENT}", headers=headers(), timeout=60)
    d = r.json() if r.status_code == 200 else {}
    print("deployment:", r.status_code, d.get("state"), (d.get("status") or {}).get("message", "")[:200])
    r = httpx.get(f"{API}/deployedModels", headers=headers(), timeout=60)
    for m in (r.json().get("deployedModels") or []) if r.status_code == 200 else []:
        print("  adapter:", m.get("model"), m.get("state"))


def down() -> None:
    r = httpx.delete(f"{API}/deployments/{DEPLOYMENT}", headers=headers(), params={"ignoreChecks": "true"}, timeout=120)
    print("delete deployment:", r.status_code, r.text[:200])


def evaluate(n: int = 80) -> None:
    load_key()
    client = OpenAI(base_url=FIREWORKS_URL, api_key=os.environ["FIREWORKS_API_KEY"])
    rng = np.random.default_rng(0)  # same selection as scripts/eval_judge.py
    test = [c for c in cases() if c["split"] == "test"]
    pick = []
    for t in ("one_object", "several_objects"):
        sub = [c for c in test if c["truth"] == t]
        pick += [sub[i] for i in rng.choice(len(sub), size=min(n // 2, len(sub)), replace=False)]
    images = {}
    prompts = []
    for c in pick:
        img = images.setdefault(c["image"], Image.open(D / "images" / f"{c['image']}.png").convert("RGB"))
        prompts.append(example(img, c["inst"], c["truth"])["messages"][:2])  # system + user, as in training
    deployment = f"{ACCOUNT}/deployments/{DEPLOYMENT}"
    results = {}
    for label, model in ((TUNED.rsplit("/", 1)[-1], f"{TUNED}#{deployment}"),):
        def run(msgs):
            try:
                r = client.chat.completions.create(model=model, messages=msgs, max_tokens=512, temperature=0)
                return _first_json(r.choices[0].message.content or "")["verdict"]
            except Exception as e:  # count as wrong, but report
                return f"error: {str(e)[:120]}"
        with ThreadPoolExecutor(8) as ex:
            verdicts = list(ex.map(run, prompts))
        per = {t: float(np.mean([v == t for v, c in zip(verdicts, pick) if c["truth"] == t])) for t in ("one_object", "several_objects")}
        errors = [v for v in verdicts if v.startswith("error")]
        results[label] = {"balanced_accuracy": float(np.mean(list(per.values()))), "recall_one": per["one_object"],
                          "recall_several": per["several_objects"], "errors": len(errors), "n": len(pick),
                          "verdicts": dict(Counter(verdicts))}
        print(f"{label:18} balanced acc {results[label]['balanced_accuracy']:.0%} (one {per['one_object']:.0%}, "
              f"several {per['several_objects']:.0%}){f'  errors {len(errors)}: {errors[0]}' if errors else ''}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "glimmer_results.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | results, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"up": up, "load": load_adapter, "status": status, "eval": evaluate, "down": down}[cmd]()
