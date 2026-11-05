# VeriCell product demo: what's real

**Final video:** `out/VeriCell-product-demo.mp4` (2:32, 1920×1080, 30 fps) · preview `out/VeriCell-product-demo-preview.mp4` (960 px) · captions `out/VeriCell-product-demo.srt`

## Footage
One continuous recording of the **live public demo** (pranavdaware1-1--vericell-web.us-east.modal.run),
captured on Nov 4 by `record/record.py`: 4,651 frames streamed from Chrome's screencast at 2× resolution
over 133 seconds. Every click, keystroke and scroll in the video happened on the live site, and every
result on screen came from that run:

- the landing page and its recorded run · Open the lab · the Colon example · the "Largest" question
- the agent's real run (53 s, shown at about 2× speed): Muse Spark's look and plan, the measured SAM 3.1 vocabulary, Cellpose-SAM 176 objects, SAM 3.1 185 objects, 531 candidates scored, Muse Spark judging #470 and #321, then asking the scientist
- the answer (8,240 px², range 2,862–23,725) and "the conclusion could change: #321 is 6% larger … 59% likely"
- Inspect #321 on the real microscopy image
- the claim test "at most 9,000" → **1-organoid fragile** (#470, 36%)
- Yes on #321 → before/after table (8,240 → 8,772, #104 → #321); No on #470, #299, #150 → "the conclusion holds" → claim **robust**
- Plan re-imaging (targets on the image) · Download evidence

## Added in the edit (Remotion)
- Camera moves and zooms, all aimed at element boxes logged during the recording
- The cursor, drawn from the recorded mouse moves and clicks (headless Chrome has no visible cursor)
- Focus rings, callouts naming each model, captions, and the scene dissolves
- Three short graphics over blurred real footage: ground truth (8,818 px², MorphoOrgaVQA annotation),
  the tool comparison (from each tool's documentation), and the benchmark numbers (RESULTS.md)
- Narration: ElevenLabs "Brian" (`../ELEVENLABS.md`); music: an instrumental loop generated with ElevenLabs sound generation, ducked under the voice; sound effects from Remotion's library. Audio normalised to -14 LUFS

## Rebuild
`python record/record.py` → `python build_timeline.py` → `npx remotion studio` (preview) or `npx remotion render VeriCellDemo out/VeriCell-product-demo.mp4`
