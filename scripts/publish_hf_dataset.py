"""Publish VeriCell's benchmark results as a Hugging Face dataset (numbers only, no source images).

The source images belong to third-party datasets (LIVECell, OrganoID, OrgaSegment, OrgaExtractor)
with their own licences, so only VeriCell's own outputs are published: per-question answers and
scores, plans, calibration reports and judge evaluations.

    .venv/bin/python scripts/publish_hf_dataset.py pranavdaware/vericell-results   # needs `hf auth login`
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "hf_dataset"

CARD = """---
license: mit
pretty_name: VeriCell benchmark results
tags:
- microscopy
- organoids
- cell-segmentation
- uncertainty
- calibration
- agents
- meta-hackathon-2026
---

# VeriCell benchmark results

Outputs of [VeriCell](https://github.com/dawarepranav/VeriCell), a microscopy agent that knows when
it's wrong, built for the Meta AI Developer Hackathon 2026 on Muse Spark, SAM 3.1, Muse Voice
Transcribe, Muse Spark Open and Muse Glimmer (Fireworks). Live demo:
https://pranavdaware1-1--vericell-web.us-east.modal.run

Numbers only: no microscopy images are included (they belong to their source datasets).

| Path | Contents |
|---|---|
| `morphoorgavqa/results.json` | Median APE per question type and method, benchmark credit, segmentation F1, both ground-truth views |
| `morphoorgavqa/records_<method>.json` | Every scored field: image, category, slot, wording, ground truth, prediction, error, credit |
| `morphoorgavqa/plans.json` | Muse Spark's JSON plan for each of the 16 benchmark questions |
| `livecell/summary.json` | Segmentation, error-detection AUROC, calibration and count-interval results on 24 LIVECell images |
| `calibration/*.json` | Split-conformal factors and leave-one-image-out coverage per question type |
| `judge/*.json` | Crop-judge accuracy: hosted Muse Spark, Muse Spark Open, fine-tuned Muse Glimmer |

Headline: on MorphoOrgaVQA (68 images x 16 questions) VeriCell has lower median error than the
published MorphoOrgaAgent on all 16 question types; on LIVECell its per-cell confidence has an
expected calibration error of 0.012. See the repository's RESULTS.md for methods and caveats.
"""


def main(repo: str) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "morphoorgavqa").mkdir(parents=True)
    for d in ("livecell", "calibration", "judge"):
        (OUT / d).mkdir()
    (OUT / "README.md").write_text(CARD)
    mo = ROOT / "outputs" / "morphoorga"
    shutil.copy(mo / "results.json", OUT / "morphoorgavqa")
    shutil.copy(mo / "plans.json", OUT / "morphoorgavqa")
    for f in sorted(mo.glob("records_official_*.json")) + sorted(mo.glob("records_debris_filtered_*.json")):
        shutil.copy(f, OUT / "morphoorgavqa" / f.name)
    shutil.copy(ROOT / "outputs" / "ui" / "summary.json", OUT / "livecell")
    for f in (ROOT / "outputs" / "models").glob("*_calibration.json"):
        shutil.copy(f, OUT / "calibration")
    for f in (ROOT / "outputs" / "judge_eval").glob("*.json"):
        shutil.copy(f, OUT / "judge" / f.name)

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo, repo_type="dataset", commit_message="VeriCell results")
    print(f"published: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pranavdaware/vericell-results")
