# VeriCell results (frozen 2026-10-28)

All numbers are out-of-fold: every image is scored by a verifier trained on other images only.
Benchmark variants were compared once on these test sets and then frozen; no further tuning on them.

## 1. MorphoOrgaVQA (organoid measurement questions)

68 brightfield organoid images × 16 questions (4 metrics × {largest/most-extreme, three-tier means} ×
{clear, open} wording). Ground truth and scoring follow the benchmark's published protocol; our
ground-truth generator matches the official one to within 3e-16 on every mask.

**Headline configuration: VeriCell v1** (Cellpose-SAM + zoom-tiled SAM 3.1 text prompts, per-cell
verifier, verified fusion, Muse Spark planner, deterministic measurement engine).

Median absolute percentage error (lower is better). Both wordings give identical numbers for VeriCell
because the planner maps them to the same plan.

| Metric | Question | MorphoOrgaAgent (published, clear / open) | Cellpose-SAM | **VeriCell v1** |
|---|---|---|---|---|
| Area | largest organoid (area + centroid) | 16.77 / 17.40 | 4.94 | **3.62** |
| Area | three-tier means | 38.74 / 36.77 | 32.58 | **28.05** |
| Perimeter | longest boundary | 15.36 / 15.36 | 9.63 | **7.33** |
| Perimeter | three-tier means | 22.03 / 22.24 | 19.54 | **16.52** |
| Roundness | roundest organoid | 1.54 / 1.56 | 1.57 | **1.38** |
| Roundness | three-tier means | 2.48 / 2.51 | 2.05 | **2.03** |
| Roughness | roughest organoid | 46.02 / 46.02 | 50.21 | **38.94** |
| Roughness | three-tier means | 31.75 / 33.74 | 28.25 | **23.80** |

- Planner (Muse Spark): right metric and operation for **16/16** questions.
- Benchmark credit (1 = within 15 %, 0.5 = within 30 %): Cellpose-SAM 0.585 → VeriCell v1 0.618.
- Sanity check: ground-truth masks through the same planner + measurement engine give 0.00 % error.

**Segmentation quality** (F1 at IoU 0.5):

| | Precision | Recall | F1 |
|---|---|---|---|
| Cellpose-SAM | 0.551 | 0.650 | 0.596 |
| VeriCell v1 | 0.774 | 0.537 | 0.634 |
| **VeriCell v3** (+ Cellpose pyramid, merged lobes, Spark crop judge) | 0.761 | 0.601 | **0.672** |

v3 finds more organoids but its "most extreme organoid" answers are less reliable (it beats the
published baseline on 14/16 questions vs 16/16 for v1), so v1 is the benchmark headline and v3 is
used for segmentation and in the interactive agent.

### Caveats
- **68 vs 69 images.** The paper doesn't list its files; we rebuilt the set from the test splits of
  OrgaSegment (12), OrgaExtractor (10) and OrganoID (46). Close to, not file-for-file identical with,
  the published evaluation.
- **Verifier trained on the same image pool** (out-of-fold by image, so no image is scored by a
  model that saw it). A fully independent check would train on the source datasets' training splits.
- **Debris in the ground truth.** OrganoID's own paper discards objects under 200 px as debris
  ("discarded if the total area was below 200 pixels", Matthews et al. 2022), but the benchmark's
  masks keep them: 427 of 2,324 OrganoID objects are under 100 px. Applying that rule to OrganoID
  ground truth and predictions alike, VeriCell v1's tier errors drop (area 28.1 → 23.3 %, perimeter
  16.5 → 12.4 %, roughness 23.8 → 19.5 %). Official numbers above are the headline.

## 2. LIVECell (dense phase-contrast cells): error detection and honest confidence

24 test images (3 per cell type, 8 types), 6,449 cells.

| | Precision | Recall | F1 |
|---|---|---|---|
| Cellpose-SAM | 0.910 | 0.809 | 0.857 |
| SAM 3.1 text, zoom-tiled | 0.797 | 0.802 | 0.799 |
| VeriCell verified fusion | 0.875 | 0.829 | 0.851 |

| Spotting wrong cells (AUROC) | |
|---|---|
| SAM 3.1's own confidence | 0.823 |
| Agreement between models alone | 0.846 |
| **VeriCell verifier** | **0.866** |

- **Calibration:** expected calibration error **0.012** ("80 % sure" ≈ right 80 % of the time).
- **Review queue:** a human checking the 12.5 % least-certain cells lifts F1 from 0.851 to **0.880**.
- **Count intervals:** 90 % split-conformal intervals contained the true count in **92 %** of images.
- On LIVECell, fusion does not beat Cellpose-SAM on F1; the gain there is knowing which cells are wrong.
- LIVECell is likely in Cellpose-SAM's training distribution (its authors note how homogeneous the
  dataset is), so the organoid benchmark is the fairer zero-shot test.

## 3. Findings along the way
- **Zooming in fixes SAM 3.1 on dense scenes.** Text-prompt F1 on 6 crowded LIVECell images: 0.27 → 0.82
  with 3×3 upscaled tiles (the 691-cell image: F1 0.06 → 0.72).
- **Cellpose-SAM splits budding organoids into lobes** at every scale; 417 of 522 "touching parts" groups
  with a clear ground-truth answer were really one organoid.
- **Roughness errors come from picking the wrong extreme organoid**, not from pixel staircasing: on
  correctly matched organoids roughness error is ~16 %, and boundary smoothing doesn't reduce it.
- **Muse Spark crop judge** ("one organoid or several?"): 74 % right on glued-together organoids, 48 % on
  true single ones (~61 % balanced) on 60 labelled cases; added as a verifier signal, small effect.

## 4. The agent: calibrated answer ranges

The live agent answers with a 90% range per answer, calibrated by split conformal prediction on
held-out images (see `vericell/answers.py`, `scripts/calibrate.py`). Leave-one-image-out coverage
for every question type (count, largest/roundest/roughest object, tier means, means):
**89–92 % on organoids (68 images) and cells (24 images)** against a 90 % target. Ranges are honest
but wide on hard images, because errors are heavy-tailed: most answers are within a few percent,
while about 1 image in 10 fails badly (usually when the true largest organoid was never segmented).

## 5. Open weights: Muse Spark Open (Fireworks) vs hosted Muse Spark

Every reasoning step (scene description, planner, crop judge, agent controller) runs on either
backend (`VERICELL_LLM=meta|open`, or the switch in the UI). Segmentation uses Cellpose-SAM (open)
and SAM 3.1 (Meta API); the verifier is our own model.

| Check | Hosted Muse Spark 1.3 | Muse Spark Open |
|---|---|---|
| Planner, 16 MorphoOrgaVQA questions (right metric + operation) | 16/16 | **16/16** |
| Planner wall time, 16 questions in parallel | 33 s | **21 s** (low reasoning effort) |
| Crop judge, 80 labelled cases (balanced accuracy) | 72 % | 70 % |
| Agent controller, same state | judges object #150 | judges object #150 (same reasoning) |
| Full agent run on a 2592×1944 organoid image | completes | completes (same actions) |

- **Reasoning effort matters per task:** low effort kept the planner at 16/16 and made it ~7× faster,
  but dropped the crop judge from 70 % to 60 %, so the judge and controller keep full reasoning.
- **Visual few-shot didn't help:** 6 labelled reference crops (from other images) changed the judge by
  −7 points (hosted) and −1 point (open).
- **Fine-tuning Muse Spark Open:** supervised LoRA is supported on Fireworks, but a fine-tuned 816B
  model needs dedicated on-demand GPUs to serve; out of scope for this budget.
- **Fine-tuned Muse Glimmer 30B crop judge (didn't beat Muse Spark):** LoRA SFT on Fireworks, trained on
  odd-index images and scored on the same 80 test crops as the hosted judges
  (`scripts/glimmer_judge.py`, served by a live-merge deployment on one H200, deleted after each eval).

  | Run | Training data | Balanced acc. | Recall "one" | Recall "several" |
  |---|---|---|---|---|
  | v1 | 2,078 crops; joined neighbour pairs as "several" | 57 % | 28 % | 88 % |
  | v2 | hard set: every joined-lobe single ×3, more plain singles | 61 % | 48 % | 75 % |
  | v3 | v2 data, LoRA rank 32, 4 epochs, lr 2e-4 | 61 % | 40 % | 82 % |
  | Muse Spark (hosted, no tuning) | none | **72 %** | | |

  v1 learned the shortcut "a joined outline means several"; harder single-object examples in v2 fixed
  part of that, and more capacity in v3 did not move it further. VeriCell keeps Muse Spark as the judge.

## 6. Does "the conclusion could change" mean anything? (fragility study)

`scripts/fragility_study.py`: all 68 MorphoOrgaVQA images x 4 "most extreme organoid" questions,
out-of-fold verifier probabilities, the agent's own verification rules (set before this study, not
tuned on it). "Wrong" = more than 15% error, the benchmark's full-credit threshold. The human check
is simulated with ground truth: each object VeriCell names is resolved as real if it matches a
ground-truth organoid.

| Question | Wrong | Wrong when VeriCell says **holds** | Wrong when it says **could change** | Wrong ones the warning catches | Fixed by checking the named objects |
|---|---:|---:|---:|---:|---:|
| Largest area | 25/68 | **13%** (3/23) | 49% (22/45) | 88% | 7 (1.4 objects each), 0 broken |
| Longest perimeter | 27/68 | 30% (6/20) | 44% (21/48) | 78% | 7 (1.5 objects each), 0 broken |
| Roundest | 1/68 | 0% (0/1) | 1% (1/67) | 100% | 1 |
| Roughest edge | 50/68 | 67% (10/15) | 75% (40/53) | 80% | 2 (and 2 broken) |

- **Size conclusions (largest area, longest perimeter): the verdict works.** When VeriCell says the
  conclusion holds it is wrong 21% of the time (median error 4%); when it says it could change, 46%
  (median error 12%). The warning catches 83% of wrong size conclusions, and checking about 1.5
  named objects fixes 14 of the 52 without breaking a right one.
- **Roundest: over-cautious.** Roundness values cluster, so near ties fire on almost every image,
  while the answer is almost never wrong.
- **Roughest edge: not fixable by checking objects.** Its errors come from measuring the boundary
  (section 3), not from picking the wrong organoid, so neither the warning nor a check helps.

## 7. SAM 3.1 vocabulary: measured prompts instead of guessed ones

`scripts/build_sam_vocab.py`: 14 organoid and 10 cell phrases screened against ground truth on
training images (even index per source dataset); the best pair per image type is tested on the
held-out half against the phrases Muse Spark picked on its own. The agent now starts SAM 3.1 from
this library (`vericell/sam_vocab.json`) and falls back to Muse Spark's words for other images.

| Held-out images | Muse Spark's phrases | Library | SAM "lost" (under 30% of Cellpose's count) |
|---|---:|---:|---|
| Organoids, 34 images ("small round blobs" + "dark cell clump") | F1 0.441 | **0.575** | 10 → **1** images |
| · OrgaExtractor (dense colon), 5 | 0.295 | **0.553** | |
| · OrgaSegment (intestinal), 6 | 0.376 | **0.637** | |
| · OrganoID, 23 | 0.490 | **0.564** | |
| Cells, 8 LIVECell images ("cell" + "cells") | 0.557 | **0.576** | 1 → 1 |

The word "organoid" alone scores F1 0.000 on the screening images (SAM returns nothing), and Muse
Spark chose it first on almost every image; "round organoid" scores 0.556. The benchmark numbers in
section 1 were frozen with Muse Spark's phrases and are not re-run with the library.

## Reproduce
```bash
.venv/bin/python scripts/get_livecell.py --per-type 3
.venv/bin/python scripts/run_candidates.py && .venv/bin/python scripts/build_results.py
.venv/bin/python scripts/get_morphoorga.py
.venv/bin/python scripts/run_morphoorga.py
.venv/bin/python scripts/build_sam_vocab.py && .venv/bin/python scripts/fragility_study.py
```
