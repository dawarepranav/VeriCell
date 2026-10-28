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

## Reproduce
```bash
.venv/bin/python scripts/get_livecell.py --per-type 3
.venv/bin/python scripts/run_candidates.py && .venv/bin/python scripts/build_results.py
.venv/bin/python scripts/get_morphoorga.py
.venv/bin/python scripts/run_morphoorga.py
```
