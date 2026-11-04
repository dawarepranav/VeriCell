# VeriCell architecture and agent harness

VeriCell answers questions about microscopy images and says how sure it is. A fixed backbone does
the heavy work, Muse Spark decides only where judgement is needed, and deterministic code computes
every number.

```
You: image + question (typed, or spoken via Muse Voice Transcribe)
        │
        ▼
┌─ Backbone (fixed order) ──────────────────────────────────────────────────┐
│ Look       Muse Spark describes the image, names objects in SAM's words  │
│ Plan       Muse Spark turns the question into a strict JSON plan         │
│ Segment    Cellpose-SAM at 1, 1/3, 1/8 scale + joined touching parts     │
│            SAM 3.1 text prompts on zoomed tiles + zoomed box re-checks   │
│ Verify     gradient-boosted verifier, 22 signals -> calibrated P(real)   │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Agent loop (Muse Spark decides, the harness enforces) ──────────────────┐
│ Controller -> Policy gate -> Tool -> Evaluator -> (next step | stop)     │
│ tools: judge_objects · try_sam_prompt · ask_human · finish               │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Answer (deterministic) ──────────────────────────────────────────────────┐
│ Measure (exact geometry) -> calibrated 90% range -> review queue -> you   │
└───────────────────────────────────────────────────────────────────────────┘
One switch picks the reasoning model: Muse Spark (Meta API) or Muse Spark Open (Fireworks).
```

## Components (code map)

| Module | What it does |
|---|---|
| `vericell/llm.py` | One interface to the reasoning model; `meta` (Muse Spark 1.3) or `open` (Muse Spark Open on Fireworks), per-task reasoning effort, lenient JSON |
| `vericell/planner.py` | Scene description and question-to-plan (strict JSON schemas) |
| `vericell/sam.py` | SAM 3.1 client: text and box prompts, zoom-tiling, parallel requests, mask decoding |
| `vericell/proposers.py` | Cellpose-SAM, its resolution pyramid, joined touching parts |
| `vericell/features.py` | 22 verification signals per object (agreement, stability, shape, appearance, context, judge) |
| `vericell/verifier.py` | Calibrated gradient boosting, out-of-fold evaluation, fusion |
| `vericell/judge.py` | Muse Spark crop judge: one object / several / not an object |
| `vericell/measure.py` | Measurement engine matching MorphoOrgaVQA's formulas exactly |
| `vericell/answers.py` | Point answers and split-conformal 90% ranges |
| `vericell/agent.py` | The agent harness (below) |
| `vericell/web/` | FastAPI app and the single-page UI |

## Harness engineering

| Principle | How VeriCell applies it |
|---|---|
| Fixed backbone, agency only where needed | Look, plan, segment, verify always run in order; the model decides only in the repair loop |
| Typed, small toolset | 4 tools with strict JSON schemas: `judge_objects` (max 6 ids), `try_sam_prompt` (1-5 words), `ask_human` (max 5 ids), `finish` |
| Policy gate | Rejects unknown tools, ids that don't exist, objects already judged, phrases already tried, calls over budget |
| Hard budgets | 6 controller steps, 18 crop-judge calls, 2 SAM re-prompts, 240 s per run |
| Evaluator after every action | Answer and range recomputed; stop when the range is within ±10%, no object could move the answer by 2%, two actions don't narrow it, or a budget runs out |
| Deterministic fallback | Controller error or invalid call: re-prompt SAM if it found under 30% of Cellpose's objects, else judge the top unjudged objects, else finish |
| Compact state | The controller sees question, plan, answer, range, top 8 uncertain objects, phrases tried, budget left; never raw masks |
| Full trace | Every phase, tool call, reason and effect streams to the UI and is saved to `outputs/runs/<run id>.json` |
| Human as a tool | `ask_human` puts specific objects in the review queue as crops, ranked by how far each could move the answer |
| Numbers never come from the model | Measurement, ranges and review ranking are deterministic code |

Design references: Anthropic, "Building effective agents"; Agentic-J (arXiv 2606.02080); The AI
Neuroscientist (arXiv 2609.25254); How spatial biologists direct and verify AI-assisted analyses
(arXiv 2609.28723).

## Answer ranges

Point answers come from the verified objects (only objects with P ≥ 0.7 can win "most extreme"
questions; tiers, counts and means weight every object by its probability). Each answer gets a 90%
range: Monte Carlo over the objects' probabilities, widened by a split-conformal factor learned per
question type on held-out images. Leave-one-image-out coverage is 89-92% for every question type
(`scripts/calibrate.py`). Reviewing objects narrows the range, because their doubt is removed.
