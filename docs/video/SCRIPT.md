# VeriCell: 3-minute demo video script

**One story, told once:** a scientist asks a plain question, VeriCell answers it honestly, warns that
the conclusion is one organoid away from wrong, the scientist checks four objects, and the answer
lands within 1% of the hand-measured truth.

**Rules for this video**
- Everything that shows VeriCell working is a **real screen recording** of the live demo
  (`RECORDING.md`). Runway is used only for cinematic b-roll and the opening mood (`RUNWAY.md`).
  Generated footage never stands in for a result.
- Every number below was checked on the live demo on Nov 4 (the "Colon" example,
  `orgaextractor_30`). Ground truth comes from the MorphoOrgaVQA hand annotations.
- Hackathon provenance inside the first 15 seconds. Meta model calls visible on screen. Benchmarks
  before 2:45. Captions on throughout (`captions.srt`).

Legend: **REC** = screen recording · **CARD** = rendered card in `cards/` · **RW** = Runway b-roll ·
**LT** = lower-third overlay in `cards/` (transparent PNG)

| Time | Picture | Narration (voice-over) | On-screen text / overlays |
|---|---|---|---|
| 0:00–0:07 | **RW-1** microscope objective over a well plate, slow push-in → hard cut to **REC** still of the Colon image with outlines | "AI can outline every organoid in a microscope image in seconds. It also gets about one in five wrong, and never tells you." | `cards/01-hook.png` overlaid at 0:03: "The best general segmenter we tested misses or mis-draws ~1 in 5 cells. No warning." |
| 0:07–0:15 | **CARD** `02-provenance.png` | "This is VeriCell, built during the Meta AI Developer Hackathon on Muse Spark, SAM 3.1, Muse Voice and Muse Spark Open. Microscopy AI that knows when it's wrong." | (card carries the text) |
| 0:15–0:30 | **REC** landing page → "Open the lab" → click the **Colon** tile → hold the mic and say the question | "A scientist asks a plain question, out loud if they like: which organoid is the largest?" | **LT** `lt-voice.png` while the mic is held |
| 0:30–1:00 | **REC** the run: stepper ticks Look → Plan → Find → Check → Decide; model chips count up; feed shows "SAM 3.1 prompts from the measured vocabulary", "Cellpose-SAM found 176", "SAM 3.1 found 185", "Scored all 531", then "Decision: look closely at 2 objects" and "ask you to check" | "Muse Spark reads the image and turns the question into a strict plan. SAM 3.1 and Cellpose-SAM outline 531 candidates, and SAM starts from words we measured it actually understands: the word 'organoid' alone finds nothing. Every candidate gets a calibrated chance of being real, and plain code measures everything. The model never invents a number. Then the agent decides what to double-check: Muse Spark judges two doubtful crops and hands them to the scientist." | **LT** `lt-muse-spark.png` (0:31), `lt-sam.png` (0:38), `lt-code.png` (0:46), `lt-agent.png` (0:53) |
| 1:00–1:20 | **REC** answer card; slow zoom (in edit) onto "The conclusion could change"; click **Inspect #321**; the image highlights it | "The answer: 8,240 square pixels, with an honest range. But VeriCell also checks whether the conclusion itself could change. Organoid 321 is six percent bigger, and only fifty-nine percent likely to be real. If it is, it's the answer." | Highlight box on the sentence "#321 is 6% larger than #104 but only 59% likely…" |
| 1:20–1:35 | **REC** "Stress-test a claim": set **at most 9,000**, click **Test it** → amber badge **1-organoid fragile** | "Say you want to publish: the largest organoid is at most 9,000. VeriCell stress-tests the claim. One organoid, thirty-six percent likely, would break it. This conclusion is one cell away from wrong." | Zoom on the badge |
| 1:35–2:00 | **REC** review queue: **Yes** on #321 → before/after table; then **No** on #470, #299, #150 as each is shown → "The conclusion holds"; re-test the claim → green **Robust** | "So the scientist checks only the objects that matter. 321 is real: the answer moves, and the table shows exactly what changed. The other three are debris. Four clicks, and the conclusion holds. Test the claim again: robust." | Zoom on the before/after table: 8,240 → 8,772, #104 → #321 |
| 2:00–2:10 | **CARD** `04-truth.png` | "Against the hand-measured truth, 8,818, the checked answer is within one percent. Before the checks it was seven percent off." | (card) |
| 2:10–2:16 | **REC** click **Plan re-imaging** (targets appear on the image) → hover **Download evidence (JSON)** | "It can tell a microscope where to look again, and export every decision as evidence." | |
| 2:16–2:45 | **CARDS** `05a` → `05b` → `05c` → `05d` (≈7 s each) | "This isn't one lucky image. On a sixty-eight-image organoid benchmark, VeriCell beats the published agent on all sixteen question types. Its warning means something: when it says a size conclusion holds, it's wrong thirteen percent of the time; when it warns, forty-nine. It spots wrong cells better than SAM's own confidence, with a calibration error of 0.012. And the whole reasoning layer runs on open weights, Muse Spark Open, with matching results." | (cards) |
| 2:45–3:00 | **RW-4** lab bench at dusk (5 s) → **CARD** `07-close.png` | "AI that measures is easy. AI that knows when it might be wrong, and helps a scientist find out, is VeriCell. Try it live." | URL + GitHub on the card |

**Running length:** narration is about 410 words, which is 2:45 at a calm 150 words per minute,
leaving room to breathe. If it runs long, cut the 2:10–2:16 beat first.

## The three moments judges must remember

1. **"The conclusion could change."** Not "this mask is uncertain": *this answer* could flip, and here is the object.
2. **"One cell away from wrong."** The claim test turns uncertainty into a sentence a reviewer understands.
3. **"Within 1% of the truth after four clicks."** Human and AI together, checked against hand annotation.

## Numbers used, and where each comes from

| On screen | Value | Source |
|---|---|---|
| Hook | ~1 in 5 cells missed or mis-drawn | Cellpose-SAM recall 0.81 on our LIVECell test (RESULTS.md §2) |
| Run | 176 / 185 / 531 objects; judge [470, 321]; 8,240 px² (2,862–23,725) | live demo, Colon example |
| Verdict | #321 6% larger, 59% likely | live demo |
| Claim | ≤ 9,000 px² → 1-organoid fragile (#470, 36%) → robust after checks | live demo |
| After checks | 8,772 px², #104 → #321 | live demo |
| Truth | 8,818 px²: within 1% after, 7% off before | MorphoOrgaVQA hand annotation |
| Benchmark | 16/16 question types beat the published agent | RESULTS.md §1 |
| Warning | wrong 13% when it holds vs 49% when it warns (largest area) | RESULTS.md §6 |
| Error detection | AUROC 0.866 vs 0.823; ECE 0.012 | RESULTS.md §2 |
| Vocabulary | "organoid" F1 0.00; SAM lost on 10 → 1 of 34 images | RESULTS.md §7 |
| Open weights | Muse Spark Open: 16/16 plans, same agent decisions | RESULTS.md §5 |

## 30-second cut (for X)

0:00–0:05 hook line over the Colon image · 0:05–0:12 "The conclusion could change: #321…" ·
0:12–0:20 claim test → "1-organoid fragile" · 0:20–0:27 four checks → "holds", "within 1% of the truth" ·
0:27–0:30 close card with URL.
