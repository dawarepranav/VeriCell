# VeriCell demo video kit

| File | What it is |
|---|---|
| `SCRIPT.md` | The 3-minute script: timecodes, picture, narration, overlays, the numbers and their sources, a 30-second cut |
| `RECORDING.md` | Click-by-click screen-recording plan for the live demo, the pre-flight checklist, and edit notes |
| `RUNWAY.md` | Runway prompts for the b-roll (opening, optional transitions, closing) and what they must never show |
| `narration.txt` | Voice-over only, by scene (read it yourself or give it to a TTS voice) |
| `captions.srt` | Caption template timed to the script (re-time after recording the voice-over) |
| `cards/` | 1920×1080 PNGs: hook overlay, provenance, ground truth, four proof cards, close; five transparent lower-thirds |
| `stills/colon-raw.png` | The Colon example image, for a slow zoom behind the hook line |
| `cards.html`, `render_cards.py` | Sources for the cards (edit, then re-render) |

**Order of work:** generate RW-1 and RW-4 in Runway → record blocks A–G → record the voice-over
from `narration.txt` → assemble on the `SCRIPT.md` timeline with the cards → captions → export.
