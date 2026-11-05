"""Capture the real VeriCell flow from the live demo for the video: 2x screenshots of every state, plus
the on-screen boxes of the elements the camera zooms to.

    python docs/video/capture.py [BASE_URL] [OUT_DIR]
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://pranavdaware1-1--vericell-web.us-east.modal.run"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "build" / "shots"
OUT.mkdir(parents=True, exist_ok=True)
meta = {}

RECT = """(sels) => Object.fromEntries(sels.map(s => { const e = document.querySelector(s); if (!e) return [s, null];
  const r = e.getBoundingClientRect(); return [s, [r.left, r.top, r.width, r.height]]; }))"""


def shot(pg, name, sels=()):
    pg.wait_for_timeout(350)
    pg.screenshot(path=OUT / f"{name}.png")
    meta[name] = pg.evaluate(RECT, list(sels))
    print("shot", name, flush=True)


def to_top(pg, sel, offset=86):
    pg.evaluate(f"(() => {{ const e = document.querySelector({json.dumps(sel)}); "
                f"window.scrollTo(0, e.getBoundingClientRect().top + scrollY - {offset}); }})()")
    pg.wait_for_timeout(500)


with sync_playwright() as p:
    try:
        b = p.chromium.launch(channel="chrome")
    except Exception:
        b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=2, color_scheme="dark")
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.wait_for_timeout(6000)  # the recorded run draws itself
    shot(pg, "01-landing", [".hero h1", ".specimen"])
    pg.evaluate("document.querySelector('#impact').scrollIntoView()")
    pg.wait_for_timeout(700)
    shot(pg, "02-landing-impact", ["#impact"])

    pg.goto(BASE + "/app", wait_until="networkidle")
    pg.wait_for_timeout(2500)
    pg.evaluate("pickExample('orgaextractor_30')")
    pg.click('.qx[data-q^="Which organoid is the largest"]')
    to_top(pg, ".askgrid")
    shot(pg, "03-ask", [".askleft", "#examples", "#chips", "#question", "#askbtn", "#modelchips"])

    pg.evaluate("() => { runAsk(); }")  # start, don't wait: frames are taken while it runs
    k = 0
    while True:
        pg.wait_for_timeout(1100)
        k += 1
        to_top(pg, ".askgrid")
        shot(pg, f"04-run-{k:02d}", [".stepcard", ".askright", "#modelchips", "#stagewrap"])
        if pg.evaluate("ask.started && !ask.running && !!ask.session") or k > 80:
            break
    pg.wait_for_timeout(800)

    to_top(pg, "#answercard")
    shot(pg, "05-answer", ["#answercard", "#answercard .a", "#answercard .verify", "#answercard .verify li",
                           "#answercard .rangebar"])
    ch = pg.evaluate("(ask.result.verification.challengers || [])[0]?.id ?? null")
    meta["_challenger"] = ch
    if ch is not None:
        pg.evaluate(f"inspect({ch})")
        pg.wait_for_timeout(900)
        to_top(pg, ".card.stage")
        shot(pg, "06-inspect-image", ["#stagewrap", f'#askview polygon[data-id="{ch}"]'])

    # stress-test a claim
    to_top(pg, "#answercard")
    pg.select_option("#aop", "at_most")
    pg.fill("#athr", "9000")
    pg.evaluate("runAudit()")
    pg.wait_for_timeout(1500)
    to_top(pg, "#answercard .audit", 300)
    shot(pg, "07-claim-fragile", [".audit", "#aout", "#aout .verdict"])

    # review queue, then the four checks
    to_top(pg, "#reviewcard")
    shot(pg, "08-review", ["#reviewcard", f'#reviewcard button[data-id="{ch}"][data-real="1"]'])
    if ch is not None:
        pg.evaluate(f"reviewAsk({ch}, true)")
        pg.wait_for_timeout(1500)
        to_top(pg, "#answercard")
        shot(pg, "09-whatif", ["#answercard", "table.whatif", "#answercard .verify"])
        nos = []
        for _ in range(5):
            nxt = pg.evaluate("(ask.result.verification.challengers || [])[0]?.id ?? null")
            if nxt is None:
                break
            nos.append(nxt)
            pg.evaluate(f"reviewAsk({nxt}, false)")
            pg.wait_for_timeout(1300)
        meta["_nos"] = nos
        to_top(pg, "#answercard")
        shot(pg, "10-holds", ["#answercard", "#answercard .verify", ".checks"])
        pg.fill("#athr", "9000")
        pg.evaluate("runAudit()")
        pg.wait_for_timeout(1500)
        to_top(pg, "#answercard .audit", 300)
        shot(pg, "11-robust", [".audit", "#aout", "#aout .verdict"])
        meta["_after"] = pg.evaluate("ask.result.text")

    # assistant features
    pg.evaluate("planReimage()")
    pg.wait_for_timeout(1500)
    to_top(pg, ".card.stage")
    shot(pg, "12-reimage", ["#stagewrap"])
    to_top(pg, "#answercard .tools", 500)
    shot(pg, "13-tools", [".tools", ".reimage"])
    pg.evaluate("window.scrollTo(0,0)")
    pg.wait_for_timeout(400)
    to_top(pg, ".askgrid")
    shot(pg, "14-left", [".askleft", "#examples", ".upload", "#llmpick", "#mic"])
    pg.click('#llmpick button[data-b="open"]')
    pg.wait_for_timeout(400)
    shot(pg, "15-open-weights", ["#llmpick"])
    b.close()

(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("done", meta.get("_challenger"), meta.get("_nos"), meta.get("_after"))
