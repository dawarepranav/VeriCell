# Screen recording plan (the real footage)

Record each block as its own clip; the editor stitches them under the narration. Total live footage
is about 2 minutes. Everything happens on the public demo:
https://pranavdaware1-1--vericell-web.us-east.modal.run

## Before recording (10 minutes)

- [ ] Chrome, a fresh profile or a guest window, no extensions or bookmarks bar; macOS Do Not Disturb on.
- [ ] Window exactly 1920×1080 (or record a 1920×1080 region); page zoom 110%.
- [ ] System appearance **dark** (the lab follows it and matches the cards).
- [ ] Open the demo once and wait about 30 s, so the cold start isn't in your footage.
- [ ] **Rehearsal run:** Colon → "Largest" → Ask VeriCell. Confirm you see "#321 is 6% larger than
      #104 but only 59% likely…". The crop judge is a live model, so if a run differs, reload and
      run again; then reload once more before the real take, so each take starts clean.
- [ ] Allow the microphone for the site (for the Muse Voice beat). A quiet room.
- [ ] Recorder: QuickTime (File → New Screen Recording) or OBS at 60 fps, cursor highlighting on.

## Block A · landing → question (0:15–0:30 in the edit), about 20 s

1. Start on the landing page `/`, scrolled to the top. Pause 2 s on the hero.
2. Click **Open the lab**.
3. Click the **Colon** tile (first in the Organoids group).
4. **Hold** the mic button and say clearly: *"Which organoid is the largest, and where is it?"*
   Release. (If transcription stumbles, click the **Largest** chip instead; it fills the same question.)

## Block B · the run (0:30–1:00), about 45 s real time

5. Click **Ask VeriCell**. Don't touch anything; let the stepper and the feed play.
6. Keep the agent feed (right column) and the model chips in frame. Expect, in order:
   "Muse Spark sees: organoids" → "SAM 3.1 prompts, from measured results" → "Cellpose-SAM found 176" →
   "SAM 3.1 found 185" → "Scored all 531" → "Decision: look closely at 2 objects" → verdicts →
   "Decision: ask you to check a few objects" → "Answer ready".
   The editor speeds this up to about 30 s; leave it unedited in the recording.

## Block C · the verdict (1:00–1:20), about 20 s

7. Scroll so the **answer card** fills the upper half: 8,240 px², the range bar, and
   **"The conclusion could change"** with its bullets.
8. Hover over the first bullet for 2 s (zoom target in the edit), then click **Inspect #321**:
   the image highlights it and the review queue scrolls into view.

## Block D · stress-test a claim (1:20–1:35), about 15 s

9. Back in the answer card, under **Stress-test a claim**: choose **at most**, type **9000**, click **Test it**.
10. Hold 3 s on the amber badge **1-organoid fragile** and its sentence (#470, 36% likely real).

## Block E · four checks (1:35–2:00), about 30 s

11. In **Help it be sure**, find **#321** (tagged "could change which organoid", "if yes → 8,772"):
    click **Yes, real**. Hold 3 s on the **before/after table** (8,240 → 8,772, #104 → #321).
12. Click **No** on **#470**. The next flagged object appears: click **No** on **#299**, then on **#150**.
    (Use the tags and ids: the queue order can shift as each check is applied.)
13. The verdict turns green: **"The conclusion holds"**. Scroll to the claim row and click **Test it**
    again: green **Robust**. Hold 3 s.

## Block F · re-imaging and evidence (2:10–2:16), about 8 s

14. Click **Plan re-imaging** (dashed target rings appear on the image) and hover
    **Download evidence (JSON)**. You don't need to download.

## Optional block G · open weights (for 05d), about 6 s

15. Scroll to **Reasoning model** and click **Muse Spark Open**. Just the toggle; no second run is needed.

## If something goes wrong

- A different verdict (no #321): reload the page and run again. Each run is fresh.
- "The public demo is busy": the hourly cap (40 questions) was hit. Wait, or run locally with
  `make serve` (same app, your own key).
- Keep one good full take as a backup before re-recording any single block.

## Edit notes

- Speed ramp Block B to about 30 s; never speed up the answer or the before/after table.
- Zooms: the verdict bullet (C), the fragile badge (D), the before/after table (E). 110–130%, 400 ms ease-out.
- Lower-thirds from `cards/lt-*.png` at the times in `SCRIPT.md`, 3–4 s each, 200 ms fade.
- Captions: `captions.srt`. Re-time it once your voice-over is recorded (the template assumes ~156 wpm).
- Music: low, neutral, no drops; duck it under the voice by about 18 dB.
- Hard limit 3:00. Export 1080p, H.264, 16 Mbps.
