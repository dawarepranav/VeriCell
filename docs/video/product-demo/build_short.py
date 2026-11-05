"""The 30-second cut for X: same real recording, its own narration. Writes src/timeline_short.json.
    python build_short.py voice   # ElevenLabs clips (needs ELEVENLABS_API_KEY in the repo's .env)
    python build_short.py         # timeline
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
LINES = {
    "s_hook": "AI misses about one in five cells, and never tells you.",
    "s_verdict": "VeriCell checks whether the conclusion itself could change. Organoid three twenty-one could take over.",
    "s_claim": "Test the claim you'd publish. It's one organoid away from wrong.",
    "s_checks": "Four human clicks. The conclusion holds, within one percent of the hand-measured truth.",
    "s_close": "VeriCell. AI that knows when it might be wrong.",
}


def voice():
    import httpx
    key = next(l.split("=", 1)[1].strip().strip('"') for l in (ROOT / ".env").read_text().splitlines() if l.startswith("ELEVENLABS_API_KEY="))
    for sid, text in LINES.items():
        r = httpx.post("https://api.elevenlabs.io/v1/text-to-speech/nPczCjzI2devNBz1zQrb", headers={"xi-api-key": key}, timeout=120,
                       params={"output_format": "mp3_44100_128"},
                       json={"text": text, "model_id": "eleven_multilingual_v2",
                             "voice_settings": {"stability": 0.38, "similarity_boost": 0.75, "style": 0.4, "use_speaker_boost": True, "speed": 1.12}})
        r.raise_for_status()
        (HERE / "public" / "audio" / f"{sid}.mp3").write_bytes(r.content)
        print(sid, "ok")


def timeline():
    spec = importlib.util.spec_from_file_location("bt", HERE / "build_timeline.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    bt.vo_text = lambda sid: LINES[sid]
    bt.TAIL = 0.45
    M, rect, grow, Scene = bt.MARK, bt.rect, bt.grow, bt.Scene
    ch = next(e["id"] for e in bt.EV if e["kind"] == "challenger")
    poly = rect("inspect", f'#askview polygon[data-id="{ch}"]')
    yes = next(c for c in bt.CLICKS if f'data-id="{ch}"' in c["target"])
    review_box = rect("review", "#reviewcard")
    scenes = [
        Scene("s_hook", "hook", fade=0.01, sfx=[{"at": 0.1, "file": "whoosh.wav", "vol": 0.18}]),
        Scene("s_verdict", ranges=[(M["answer"]["t"], M["inspect"]["t"] + 1.6)],
              cams=[{"at": 0, "rect": rect("answer", "#answercard .verify"), "fill": 0.84, "dur": 0.01},
                    {"src": M["inspect"]["t"] - 0.3, "rect": grow(poly, 9), "fill": 0.9, "dur": 0.7}],
              rings=[{"at": 0.5, "dur": 3.0, "rect": rect("answer", "#answercard .verify li")}],
              sfx=[{"at": 0.5, "file": "switch.wav", "vol": 0.2}]),
        Scene("s_claim", ranges=[(M["claim_fragile"]["t"] - 3.2, M["claim_fragile"]["t"] + 1.8)],
              window={"x": 80, "y": 170, "w": 1140},
              panel={"t": 0.2, "x": 1290, "w": 560, "kicker": "STRESS-TEST A CLAIM", "title": "“The largest organoid is at most 9,000 px².”",
                     "lines": [{"text": "1-organoid fragile", "big": True, "color": "amber"}, {"text": "One uncertain object breaks it."}]},
              cams=[{"at": 0, "rect": rect("claim_fragile", ".audit"), "fill": 0.86, "dur": 0.01}],
              rings=[{"src": M["claim_fragile"]["t"], "dur": 1.6, "rect": rect("claim_fragile", "#aout .verdict")}]),
        Scene("s_checks", ranges=[(yes["t"] - 1.2, M["whatif"]["t"] + 2.2), (M["whatif"]["t"] + 2.2, M["robust"]["t"] + 1.2)], rates=[1.5, None], extra=2.5,
              cams=[{"at": 0, "rect": [review_box[0], yes["y"] - 140, review_box[2], 300], "fill": 0.92, "dur": 0.01},
                    {"src": M["whatif"]["t"] - 0.2, "rect": rect("whatif", "table.whatif"), "fill": 0.8, "dur": 0.6},
                    {"src": M["robust"]["t"] - 0.4, "rect": rect("robust", ".audit"), "fill": 0.86, "dur": 0.5}],
              rings=[{"src": M["whatif"]["t"] + 0.2, "dur": 1.5, "rect": rect("whatif", "table.whatif"), "color": "#5cc28a"},
                     {"src": M["robust"]["t"], "dur": 1.2, "rect": rect("robust", "#aout .verdict"), "color": "#5cc28a"}],
              callouts=[{"src": M["whatif"]["t"] + 0.2, "dur": 2.0, "pos": "top", "title": "8,240 → 8,772 px²", "sub": "hand-measured truth: 8,818 · within 1%", "color": "#5cc28a"}],
              sfx=[{"src": M["robust"]["t"], "file": "ding.wav", "vol": 0.28}]),
        Scene("s_close", ranges=[(M["close"]["t"] - 3.2, bt.EV[-1]["t"])], extra=1.2, cams=[{"at": 0, "rect": None}]),
    ]
    out, t = [], 0.0
    for s in scenes:
        j = s.json(t)
        out.append(j)
        t += j["dur"] - bt.OVERLAP
    total = t + bt.OVERLAP
    (HERE / "src" / "timeline_short.json").write_text(json.dumps({"total": round(total, 3), "scenes": out}, indent=1))
    print(f"total {total:.1f}s")
    for j in out:
        print(f"  {j['id']:10} +{j['dur']:5.1f}s  {[(s['src'], s['rate']) for s in j['segs']]}")


if __name__ == "__main__":
    voice() if sys.argv[1:] == ["voice"] else timeline()
