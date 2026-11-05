# ElevenLabs voice-over for the VeriCell demo

Generate **12 clips, one per scene**, and save them with exactly these names: `hook`, `pitch`,
`ask`, `agent`, `verdict`, `claim`, `checks`, `truth`, `assistant`, `compare`, `proof`, `close`
(.mp3 or .wav). The video re-times itself to each clip. Keep the total spoken time under
about 2:40 so the video stays under 3:00.

**Settings**
- Model: Eleven Multilingual v2 (most consistent for narration)
- Voice: a calm, warm, mid-pitch narrator (library examples: "Brian", "Daniel", "Alice"). Avoid "trailer" voices.
- Stability 50 · Similarity 75 · Style 10 · Speaker boost on · Speed 1.0
- Output: MP3 44.1 kHz 192 kbps, or WAV

**Pronunciation** (Pronunciation dictionary → alias, or type the spelling as written below)
- VeriCell → "Vehry-cell" (like *verify* + *cell*)
- SAM 3.1 → written below as "SAM three point one"
- napari → "nah-PAH-ree"
- organoid → "OR-guh-noid"
- MorphoOrgaAgent → written below as "Morpho-Orga-Agent"

---

## hook
AI can outline every organoid in a microscope image in seconds. <break time="0.3s" /> It gets about one in five wrong, and it never tells you. <break time="0.4s" /> Those errors end up in published results.

## pitch
This is VeriCell: an AI research assistant for microscopy, built during the Meta AI Developer Hackathon, on Muse Spark, SAM three point one, Muse Voice, and Muse Spark Open. <break time="0.4s" /> It doesn't just outline cells. It answers your scientific question, and tells you whether to trust the answer.

## ask
A biologist picks an image, and asks, in plain words: <break time="0.3s" /> which organoid is the largest?

## agent
Muse Spark reads the image, and writes a strict analysis plan. <break time="0.3s" /> SAM three point one and Cellpose-SAM outline five hundred and thirty-one candidates, starting from words we measured SAM actually understands. <break time="0.3s" /> Every object gets a calibrated chance of being real, and plain code does every measurement. <break time="0.4s" /> Then the agent decides what to double-check. Muse Spark judges two doubtful crops, and hands them to the scientist.

## verdict
Here is the answer, with an honest range. <break time="0.4s" /> And here is what other tools don't tell you: the conclusion itself could change. <break time="0.3s" /> Organoid three twenty-one is six percent bigger, and only fifty-nine percent likely to be real.

## claim
Now test the claim you would publish: the largest organoid is at most nine thousand square pixels. <break time="0.4s" /> VeriCell finds it is one organoid away from wrong.

## checks
So the scientist checks only what matters. <break time="0.3s" /> Three twenty-one is real. The answer moves, and you see exactly what changed. <break time="0.3s" /> Three more are debris. <break time="0.3s" /> Four clicks, and the conclusion holds. The claim is now robust.

## truth
Against hand-measured ground truth, the answer went from seven percent off, <break time="0.3s" /> to within one percent.

## assistant
And it works like a research assistant. It plans where a microscope should look again, exports every decision as evidence, takes spoken questions, and runs on your own images.

## compare
Today, napari and CellProfiler segment and measure, but never say when to doubt the result. <break time="0.3s" /> Agents like Morpho-Orga-Agent and GenCellAgent answer questions, without calibrated confidence. <break time="0.4s" /> VeriCell does both, and checks its own conclusions.

## proof
And it is measured. <break time="0.3s" /> On a sixty-eight image organoid benchmark, it beats the published agent on all sixteen question types. <break time="0.3s" /> When it says a conclusion holds, it is wrong thirteen percent of the time. When it warns, forty-nine. <break time="0.4s" /> And the whole reasoning layer runs on open weights.

## close
VeriCell. <break time="0.4s" /> AI that knows when it might be wrong, and helps scientists find out what is true. <break time="0.3s" /> Try it live.
