# VeriCell

A microscopy agent that knows when it's wrong: it segments the cells you ask about,
checks every cell, repairs what it can, and answers with calibrated error bars plus a
short review queue.

Built for the Meta AI Developer Hackathon (Oct 27 – Nov 4, 2026). All code in this
repository was written during the hackathon; it uses Meta's models (Muse Spark, SAM 3.1, Muse Voice
Transcribe) and open-source tools (Cellpose-SAM, scikit-learn, OpenCV).

**Results: see [RESULTS.md](RESULTS.md).** On the MorphoOrgaVQA organoid benchmark, VeriCell has lower
median error than the published MorphoOrgaAgent on all 16 question types; on LIVECell its per-cell
confidence is calibrated (ECE 0.012) and spots wrong cells better than SAM 3.1's own confidence.

## Setup

```bash
python3.11 -m venv .venv   # PyTorch (for Cellpose) has no wheels for 3.15 yet
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then put your Meta Model API key in .env
```

## Smoke test

Checks that your key can reach every Meta model VeriCell uses (Muse Spark text + vision,
SAM 3.1 concept and box prompts, Muse Voice Transcribe):

```bash
.venv/bin/python scripts/smoke_test.py
```

## Baseline (LIVECell sample)

Fetches 3 test images per LIVECell cell type (24 images) using HTTP range requests, then
scores SAM 3.1 with a text prompt, Cellpose-SAM, and the hybrid (Cellpose boxes re-segmented
by SAM 3.1). Predictions are cached in `outputs/cache/`.

```bash
.venv/bin/python scripts/get_livecell.py --per-type 3
.venv/bin/python scripts/baseline.py
```

## Organoid benchmark (MorphoOrgaVQA)

```bash
.venv/bin/python scripts/get_morphoorga.py    # rebuilds the 68-image set from OrgaSegment, OrgaExtractor, OrganoID
.venv/bin/python scripts/run_morphoorga.py    # plan -> segment -> verify -> measure -> answer, then score
```

## Web UI

```bash
.venv/bin/uvicorn vericell.web.app:app --port 8765
```
