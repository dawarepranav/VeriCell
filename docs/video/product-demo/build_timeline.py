"""Turn the real recording (record/frames + events.json) into public/footage.mp4 and src/timeline.json.

Every camera target, ring and cursor position comes from the recording's event log: the on-screen
boxes of real UI elements and the real mouse moves and clicks.
    python build_timeline.py
"""

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "record"
PUB = HERE / "public"
LEAD, TAIL, OVERLAP = 0.25, 0.55, 0.3

EV = json.loads((REC / "events.json").read_text())
FR = json.loads((REC / "frames.json").read_text())
MARK = {e["label"]: e for e in EV if e["kind"] == "mark"}
CLICKS = [e for e in EV if e["kind"] == "click"]
MOVES = [e for e in EV if e["kind"] == "move"]
T = lambda k, **kw: next(e for e in EV if e["kind"] == k and all(e.get(a) == b for a, b in kw.items()))["t"]  # noqa: E731


def rect(label, sel):
    return MARK[label]["rects"].get(sel)


def union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    return [x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0]


def grow(r, k):
    """Box r grown k times around its centre."""
    return [r[0] - r[2] * (k - 1) / 2, r[1] - r[3] * (k - 1) / 2, r[2] * k, r[3] * k]


def footage():
    """Screencast frames (variable rate) -> constant 30 fps MP4."""
    lst = REC / "concat.txt"
    lines = []
    for a, b in zip(FR, FR[1:] + [None]):
        d = (b["t"] - a["t"]) if b else 0.5
        lines += [f"file 'frames/{a['file']}'", f"duration {max(d, 0.001):.4f}"]
    lines.append(f"file 'frames/{FR[-1]['file']}'")
    lst.write_text("\n".join(lines))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf",
                    "scale=2560:1440:flags=lanczos,fps=30", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                    "-pix_fmt", "yuv420p", PUB / "footage.mp4"], check=True, cwd=REC)


def vo_len(sid):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 PUB / "audio" / f"{sid}.mp3"], capture_output=True, text=True).stdout)


def vo_text(sid):
    text = (HERE.parent / "ELEVENLABS.md").read_text()
    m = re.search(rf"^## {sid}\n(.+?)(?=\n## |\Z)", text, flags=re.S | re.M)
    return re.sub(r"<break[^>]*/>\s*", "", m.group(1).strip())


def captions(sid, d):
    words = vo_text(sid).split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 8 or (re.search(r"[.,:?]$", w) and len(cur) >= 4):
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    total = sum(len(c) for c in chunks)
    out, t = [], LEAD
    for c in chunks:
        dt = d * len(c) / total
        out.append({"t": round(t, 3), "end": round(t + dt, 3), "text": c})
        t += dt
    return out


class Scene:
    def __init__(self, sid, kind="footage", ranges=(), rates=None, **kw):
        self.id, self.kind, self.kw = sid, kind, kw
        self.vo = vo_len(sid)
        self.dur = self.vo + LEAD + TAIL + kw.pop("extra", 0)
        self.segs = []
        if ranges:
            # split the scene's time across the source ranges; fixed rates first, the rest share what is left
            rates = rates or [None] * len(ranges)
            fixed = sum((b - a) / r for (a, b), r in zip(ranges, rates) if r)
            free = sum(b - a for (a, b), r in zip(ranges, rates) if not r)
            rest = max(0.5, self.dur - fixed)
            t = 0.0
            for (a, b), r in zip(ranges, rates):
                r = r or max(0.3, free / rest)
                d = (b - a) / r
                self.segs.append({"from": round(t, 3), "dur": round(d, 3), "src": round(a, 3), "rate": round(r, 4)})
                t += d
            self.dur = max(self.dur, t)

    def at(self, src):
        """Scene time of a source time (None if it isn't in this scene's footage)."""
        for s in self.segs:
            if s["src"] - 1e-6 <= src <= s["src"] + s["dur"] * s["rate"] + 1e-6:
                return round(s["from"] + (src - s["src"]) / s["rate"], 3)
        return None

    def rate_at(self, src):
        return next((s["rate"] for s in self.segs if s["src"] <= src <= s["src"] + s["dur"] * s["rate"]), 1.0)

    def json(self, start):
        cams = []
        for c in self.kw.get("cams", []):
            t = c.get("at") if "at" in c else self.at(c["src"])
            if t is not None:
                cams.append({"t": max(0, t), "rect": c.get("rect"), "fill": c.get("fill", 0.85), "dur": c.get("dur", 0.6)})
        rings = []
        for r in self.kw.get("rings", []):
            t = r.get("at") if "at" in r else self.at(r["src"])
            if t is not None and r.get("rect"):
                rings.append({**{k: v for k, v in r.items() if k not in ("src", "at")}, "t": t})
        cursor, clicks, sfx = [], [], []
        for m in MOVES:
            t = self.at(m["t"])
            if t is not None:
                cursor.append({"t": t, "dur": round(m["dur"] / self.rate_at(m["t"]), 3), "x0": m["x0"], "y0": m["y0"], "x": m["x"], "y": m["y"]})
        for c in CLICKS:
            t = self.at(c["t"])
            if t is not None:
                clicks.append({"t": t, "x": c["x"], "y": c["y"]})
                sfx.append({"t": t, "src": "mouse-click.wav", "vol": 0.22})
        for x in self.kw.get("sfx", []):
            t = x.get("at") if "at" in x else self.at(x["src"])
            if t is not None:
                sfx.append({"t": t, "src": x["file"], "vol": x.get("vol", 0.3)})
        callouts = []
        for c in self.kw.get("callouts", []):
            t = c.get("at") if "at" in c else self.at(c["src"])
            if t is not None:
                callouts.append({**{k: v for k, v in c.items() if k not in ("src", "at")}, "t": t})
        out = {"id": self.id, "kind": self.kind, "start": round(start, 3), "dur": round(self.dur, 3), "vo_at": LEAD,
               "segs": self.segs, "cams": cams, "rings": rings, "cursor": cursor, "clicks": clicks, "sfx": sfx,
               "callouts": callouts, "captions": captions(self.id, self.vo), "vo_len": round(self.vo, 3)}
        for k in ("window", "panel", "lockup", "dim", "fade"):
            if k in self.kw:
                out[k] = self.kw[k]
        return out


def main():
    footage()
    click_open = CLICKS[0]["t"]
    run0, run1 = T("run_start"), T("run_end")
    ins = MARK["inspect"]
    ch = next(e["id"] for e in EV if e["kind"] == "challenger")
    poly = rect("inspect", f'#askview polygon[data-id="{ch}"]')
    yes = next(c for c in CLICKS if f'data-id="{ch}"' in c["target"])
    arun = [c for c in CLICKS if c["target"] == "#arun"]
    run_box = rect("running", ".askright")

    scenes = [
        Scene("hook", "hook", sfx=[{"at": 0.2, "file": "whoosh.wav", "vol": 0.18}]),
        Scene("pitch", ranges=[(0.0, click_open + 0.25)], lockup=[1.2, 0],
              cams=[{"at": 0, "rect": None}, {"at": 4.0, "rect": union(rect("landing", ".hero h1"), rect("landing", ".specimen")), "fill": 0.97, "dur": 1.4},
                    {"src": click_open - 1.6, "rect": None, "dur": 0.9}]),
        Scene("ask", ranges=[(click_open + 0.25, run0 + 1.2)], extra=2.0,
              cams=[{"src": MARK["lab"]["t"], "rect": grow(rect("lab", ".askleft"), 1.05), "fill": 0.95, "dur": 0.7}],
              rings=[{"src": MARK["question"]["t"], "dur": 1.6, "rect": rect("question", "#chips"), "color": "#7fd3e6"}]),
        Scene("agent", ranges=[(run0 + 1.2, run1 + 1.6)],
              cams=[{"at": 0, "rect": None},
                    {"at": 1.0, "rect": [run_box[0], run_box[1], run_box[2], run_box[3] * 0.5], "fill": 0.9, "dur": 0.8},
                    {"at": 5.2, "rect": rect("running", "#stagewrap"), "fill": 0.98, "dur": 0.9},
                    {"at": 11.6, "rect": rect("running", ".stepcard"), "fill": 0.9, "dur": 0.8},
                    {"at": 17.6, "rect": [run_box[0], run_box[1] + run_box[3] * 0.35, run_box[2], run_box[3] * 0.6], "fill": 0.9, "dur": 0.9},
                    {"at": 24.6, "rect": None, "dur": 0.9}],
              callouts=[{"at": 0.7, "dur": 4.2, "title": "Muse Spark", "sub": "reads the image · writes the analysis plan"},
                        {"at": 5.3, "dur": 6.0, "title": "SAM 3.1 + Cellpose-SAM", "sub": "531 candidates · prompts from a measured vocabulary"},
                        {"at": 11.8, "dur": 5.4, "title": "Plain code, not the model", "sub": "a calibrated P(real) and every measurement", "color": "#ece7dd"},
                        {"at": 17.8, "dur": 7.6, "title": "Muse Spark, as the agent", "sub": "judges two doubtful crops · hands them to the scientist", "color": "#e0a33a"}]),
        Scene("verdict", ranges=[(MARK["run_done"]["t"], ins["t"] + 2.4)],
              cams=[{"src": MARK["answer"]["t"], "rect": rect("answer", "#answercard"), "fill": 0.92, "dur": 0.7},
                    {"src": MARK["answer"]["t"] + 1.6, "rect": rect("answer", "#answercard .verify"), "fill": 0.84, "dur": 0.7},
                    {"src": ins["t"] - 0.3, "rect": grow(poly, 9) if poly else None, "fill": 0.9, "dur": 1.0}],
              rings=[{"src": MARK["answer"]["t"] + 2.3, "dur": 3.6, "rect": rect("answer", "#answercard .verify li")},
                     {"src": ins["t"] + 0.6, "dur": 2.6, "rect": grow(poly, 1.5) if poly else None, "round": True, "pulse": True}],
              sfx=[{"src": MARK["answer"]["t"] + 2.3, "file": "switch.wav", "vol": 0.2}]),
        Scene("claim", ranges=[(ins["t"] + 2.4, MARK["claim_fragile"]["t"] + 2.2)],
              window={"x": 80, "y": 170, "w": 1140},
              panel={"t": 0.5, "x": 1290, "w": 560, "kicker": "STRESS-TEST A CLAIM", "title": "“The largest organoid is at most 9,000 px².”",
                     "lines": [{"text": "1-organoid fragile", "big": True, "color": "amber"},
                               {"text": "If #470, only 36% likely real, is real, the answer becomes 11,976 px² and the claim fails."},
                               {"text": "VeriCell searches for the fewest uncertain objects that would break it."}]},
              cams=[{"src": MARK["claim_fragile"]["t"] - 3.0, "rect": rect("claim_fragile", ".audit"), "fill": 0.86, "dur": 0.8}],
              rings=[{"src": MARK["claim_fragile"]["t"], "dur": 2.5, "rect": rect("claim_fragile", "#aout .verdict")}],
              sfx=[{"src": MARK["claim_fragile"]["t"], "file": "switch.wav", "vol": 0.22}]),
        Scene("checks", ranges=[(MARK["review"]["t"], MARK["whatif"]["t"] + 3.0), (MARK["whatif"]["t"] + 3.0, MARK["robust"]["t"] + 2.0)],
              rates=[0.9, None], extra=1.5,
              cams=[{"src": MARK["review"]["t"], "rect": grow(rect("review", f'#reviewcard button[data-id="{ch}"][data-real="1"]'), 1.0) and
                     [rect("review", "#reviewcard")[0], yes["y"] - 140, rect("review", "#reviewcard")[2], 300], "fill": 0.92, "dur": 0.7},
                    {"src": MARK["whatif"]["t"] - 0.2, "rect": rect("whatif", "table.whatif"), "fill": 0.8, "dur": 0.8},
                    {"src": MARK["whatif"]["t"] + 3.1, "rect": rect("review", "#reviewcard"), "fill": 0.98, "dur": 0.7},
                    {"src": MARK["holds"]["t"] - 0.2, "rect": rect("holds", "#answercard .verify"), "fill": 0.8, "dur": 0.7},
                    {"src": MARK["robust"]["t"] - 0.4, "rect": rect("robust", ".audit"), "fill": 0.86, "dur": 0.6}],
              rings=[{"src": MARK["whatif"]["t"] + 0.5, "dur": 2.4, "rect": rect("whatif", "table.whatif"), "color": "#5cc28a"},
                     {"src": MARK["holds"]["t"] + 0.2, "dur": 0.8, "rect": rect("holds", "#answercard .verify"), "color": "#5cc28a"},
                     {"src": MARK["robust"]["t"], "dur": 1.9, "rect": rect("robust", "#aout .verdict"), "color": "#5cc28a"}],
              callouts=[{"src": MARK["whatif"]["t"] + 0.4, "dur": 2.6, "pos": "top", "title": "8,240 → 8,772 px²", "sub": "the answer moves to #321 · the verdict updates", "color": "#5cc28a"}],
              sfx=[{"src": MARK["whatif"]["t"] + 0.4, "file": "whoosh.wav", "vol": 0.15}, {"src": MARK["robust"]["t"], "file": "ding.wav", "vol": 0.28}]),
        Scene("truth", "graphic", ranges=[(MARK["robust"]["t"], MARK["robust"]["t"] + 1.5)], rates=[0.2],
              sfx=[{"at": 2.1, "file": "ding.wav", "vol": 0.18}]),
        Scene("assistant", ranges=[(MARK["robust"]["t"] + 1.5, MARK["evidence"]["t"] + 2.0)],
              cams=[{"src": MARK["reimage"]["t"] - 0.2, "rect": rect("reimage", "#stagewrap"), "fill": 0.98, "dur": 0.8},
                    {"src": MARK["evidence"]["t"] - 0.3, "rect": grow(rect("evidence", ".tools"), 1.6), "fill": 0.8, "dur": 0.8}],
              callouts=[{"src": MARK["reimage"]["t"] + 0.2, "dur": 3.0, "title": "A re-imaging plan", "sub": "where a microscope should look again · Pycro-Manager z-stack script"},
                        {"src": MARK["evidence"]["t"], "dur": 2.4, "title": "Evidence, exported", "sub": "question · models · answer · every correction · full trace"}]),
        Scene("compare", "graphic", ranges=[(MARK["evidence"]["t"], MARK["evidence"]["t"] + 2.0)], rates=[0.12]),
        Scene("proof", "graphic", ranges=[(MARK["evidence"]["t"] + 2.0, MARK["evidence"]["t"] + 4.0)], rates=[0.12]),
        Scene("close", ranges=[(MARK["close"]["t"] - 5.0, EV[-1]["t"])], extra=1.0,
              cams=[{"at": 0, "rect": None}]),
    ]
    # the agent scene's beats were placed on the narration's clock: keep them on it at the new voice speed
    k = vo_len("agent") / 25.4
    agent = next(s for s in scenes if s.id == "agent")
    for key in ("cams", "callouts"):
        for c in agent.kw[key]:
            c["at"] = round(c["at"] * k, 3)
            if "dur" in c and key == "callouts":
                c["dur"] = round(c["dur"] * k, 3)
    out, t = [], 0.0
    for s in scenes:
        j = s.json(t)
        if j["id"] != "hook":
            j["sfx"].append({"t": 0.0, "src": "whoosh.wav", "vol": 0.14})
        if j["id"] == "pitch":
            j["lockup"] = [1.0, j["dur"] - 3.5]
        out.append(j)
        t += j["dur"] - OVERLAP
    total = t + OVERLAP
    (HERE / "src" / "timeline.json").write_text(json.dumps({"total": round(total, 3), "scenes": out}, indent=1))
    print(f"total {total:.1f}s")
    for j in out:
        print(f"  {j['id']:10} {j['start']:6.1f} +{j['dur']:5.1f}s  segs {[(s['src'], s['rate']) for s in j['segs']]}")


if __name__ == "__main__":
    main()
