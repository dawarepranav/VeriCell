"""Record the real VeriCell flow on the live site as video: every frame Chrome paints (CDP screencast, 2x),
plus a log of real mouse moves, clicks, scrolls and on-screen element boxes, for the Remotion edit.

    python record/record.py [BASE_URL]
Outputs record/frames/*.jpg, record/frames.json (timestamps), record/events.json.
Nothing is simulated: every click is a real click on the live site, every result comes from the live agent.
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://pranavdaware1-1--vericell-web.us-east.modal.run"
HERE = Path(__file__).resolve().parent
FRAMES = HERE / "frames"
EVENTS: list[dict] = []
FRAME_LOG: list[dict] = []
MOUSE = [960.0, 600.0]

RECT = """(sels) => Object.fromEntries(sels.map(s => { const e = document.querySelector(s); if (!e) return [s, null];
  const r = e.getBoundingClientRect(); return [s, [r.left, r.top, r.width, r.height]]; }))"""


def log(kind, **kw):
    EVENTS.append({"t": time.time(), "kind": kind, **kw})
    print(f"{time.strftime('%H:%M:%S')} {kind} {kw.get('label', kw.get('target', ''))} frames={len(FRAME_LOG)}", flush=True)


async def mark(pg, label, sels=()):
    """A named moment, with the on-screen boxes of the elements the camera will frame."""
    log("mark", label=label, rects=await pg.evaluate(RECT, list(sels)))


async def move_to(pg, sel, dur=0.7, dx=0.5, dy=0.5):
    box = await pg.locator(sel).first.bounding_box()
    x, y = box["x"] + box["width"] * dx, box["y"] + box["height"] * dy
    log("move", x0=MOUSE[0], y0=MOUSE[1], x=x, y=y, dur=dur)
    steps = max(8, int(dur * 60))
    await pg.mouse.move(x, y, steps=steps)
    MOUSE[:] = [x, y]
    await asyncio.sleep(0.15)
    return x, y


async def click(pg, sel, dur=0.7, settle=0.6):
    x, y = await move_to(pg, sel, dur)
    log("click", x=x, y=y, target=sel)
    await pg.mouse.down()
    await asyncio.sleep(0.08)
    await pg.mouse.up()
    await asyncio.sleep(settle)


async def scroll_to(pg, sel, offset=86, dur=0.9):
    """Real smooth scroll so `sel` sits `offset` px below the top."""
    y0 = await pg.evaluate("scrollY")
    y1 = await pg.evaluate(f"(() => {{ const e = document.querySelector({json.dumps(sel)}); "
                           f"return Math.max(0, e.getBoundingClientRect().top + scrollY - {offset}); }})()")
    log("scroll", y0=y0, y1=y1, dur=dur)
    n = max(2, int(dur * 60))
    for i in range(1, n + 1):
        k = i / n
        e = 4 * k ** 3 if k < .5 else 1 - (-2 * k + 2) ** 3 / 2
        await pg.evaluate(f"window.scrollTo(0, {y0 + (y1 - y0) * e})")
        await asyncio.sleep(dur / n)
    await asyncio.sleep(0.25)


async def type_into(pg, sel, text):
    await click(pg, sel, settle=0.2)
    await pg.keyboard.press("Meta+A")
    log("type", target=sel, text=text)
    await pg.keyboard.type(text, delay=110)
    await asyncio.sleep(0.3)


async def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    for f in FRAMES.glob("*.jpg"):
        f.unlink()
    async with async_playwright() as p:
        try:
            b = await p.chromium.launch(channel="chrome")
        except Exception:
            b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2, color_scheme="dark")
        pg = await ctx.new_page()
        # warm the demo so the cold start never appears on camera
        await pg.goto(BASE + "/app", wait_until="networkidle")
        await asyncio.sleep(2)
        await pg.goto(BASE + "/", wait_until="networkidle")
        await asyncio.sleep(1)

        cdp = await ctx.new_cdp_session(pg)
        n = [0]

        async def on_frame(ev):
            n[0] += 1
            name = f"{n[0]:06d}.jpg"
            (FRAMES / name).write_bytes(base64.b64decode(ev["data"]))
            FRAME_LOG.append({"file": name, "t": ev["metadata"]["timestamp"]})
            try:
                await cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception as exc:
                print("ack failed:", exc, flush=True)

        cdp.on("Page.screencastFrame", lambda ev: asyncio.ensure_future(on_frame(ev)))
        SC = {"format": "jpeg", "quality": 92, "maxWidth": 2560, "maxHeight": 1440, "everyNthFrame": 1}

        async def restart():  # a new document needs a fresh screencast, or Chrome stops sending frames
            try:
                await cdp.send("Page.stopScreencast")
            except Exception:
                pass
            await cdp.send("Page.startScreencast", SC)

        await pg.goto(BASE + "/", wait_until="networkidle")
        await cdp.send("Page.startScreencast", SC)
        log("start")

        # ---- landing: the recorded run draws itself, then into the lab
        await asyncio.sleep(5.5)
        await mark(pg, "landing", [".hero h1", ".specimen"])
        await click(pg, "nav.top .btn.primary", dur=1.0, settle=0.2)
        await pg.wait_for_url("**/app", wait_until="domcontentloaded", timeout=30000)
        await pg.wait_for_selector(".ex", timeout=30000)
        await restart()
        await asyncio.sleep(1.6)
        await scroll_to(pg, ".askgrid", dur=1.0)
        await mark(pg, "lab", [".askleft", "#examples", "#chips", "#question", "#askbtn", "#modelchips"])

        # ---- the question
        await click(pg, '.ex[data-n="orgaextractor_30"]', dur=0.9, settle=0.9)
        await click(pg, '.qx[data-q^="Which organoid is the largest"]', dur=0.8, settle=0.8)
        await mark(pg, "question", ["#question", "#chips", "#askbtn"])
        await scroll_to(pg, "#askbtn", offset=760, dur=0.9)  # the left column is taller than the screen
        await mark(pg, "askbtn", ["#askbtn", "#question", "#llmpick"])
        await click(pg, "#askbtn", dur=0.8, settle=0.3)
        await scroll_to(pg, ".askgrid", dur=1.0)
        print("state after Ask:", await pg.evaluate("JSON.stringify({ex: ask.example, q: document.querySelector('#question').value, started: ask.started, running: ask.running, disabled: document.querySelector('#askbtn').disabled, top: document.elementFromPoint(%f, %f)?.outerHTML.slice(0, 120)})" % tuple(MOUSE)), flush=True)
        log("run_start")
        await move_to(pg, ".askright", dur=1.2, dx=.5, dy=.2)
        await mark(pg, "running", [".stepcard", ".askright", "#modelchips", "#stagewrap", "#feed"])
        while not await pg.evaluate("ask.started && !ask.running && !!ask.session"):
            if await pg.evaluate("!!ask.failed"):
                raise SystemExit("the run failed on the live site (rate limit?): " + await pg.evaluate("document.querySelector('#feed').innerText.slice(-300)"))
            await asyncio.sleep(0.25)
        log("run_end")
        await asyncio.sleep(1.2)
        await mark(pg, "run_done", [".stepcard", ".askright", "#modelchips", "#answercard", "#feed"])

        # ---- the answer and the verdict
        await scroll_to(pg, "#answercard", dur=1.0)
        await mark(pg, "answer", ["#answercard", "#answercard .a", "#answercard .rangebar", "#answercard .verify",
                                  "#answercard .verify li", "#answercard .verify .ids"])
        ch = await pg.evaluate("(ask.result.verification.challengers || [])[0]?.id ?? null")
        log("challenger", id=ch)
        await asyncio.sleep(2.5)
        await click(pg, f'#answercard [data-inspect="{ch}"]', dur=0.9, settle=0.2)
        await asyncio.sleep(1.4)  # inspect() scrolls the review crop into view
        await scroll_to(pg, ".card.stage", offset=70, dur=1.0)
        await mark(pg, "inspect", ["#stagewrap", f'#askview polygon[data-id="{ch}"]'])
        await asyncio.sleep(2.5)

        # ---- stress-test a claim
        await scroll_to(pg, "#answercard .audit", offset=330, dur=1.0)
        await pg.select_option("#aop", "at_most")
        await type_into(pg, "#athr", "9000")
        await click(pg, "#arun", dur=0.6, settle=1.6)
        await mark(pg, "claim_fragile", [".audit", "#aout", "#aout .verdict"])
        await asyncio.sleep(2.0)

        # ---- the human checks
        await scroll_to(pg, "#reviewcard", dur=1.0)
        await mark(pg, "review", ["#reviewcard", f'#reviewcard button[data-id="{ch}"][data-real="1"]'])
        await asyncio.sleep(1.5)
        await click(pg, f'#reviewcard button[data-id="{ch}"][data-real="1"]', dur=1.0, settle=1.2)
        await scroll_to(pg, "#answercard", dur=1.0)
        await mark(pg, "whatif", ["#answercard", "table.whatif", "#answercard .a"])
        await asyncio.sleep(3.0)
        nos = []
        for _ in range(5):
            nxt = await pg.evaluate("(ask.result.verification.challengers || [])[0]?.id ?? null")
            if nxt is None:
                break
            nos.append(nxt)
            await scroll_to(pg, f'#reviewcard button[data-id="{nxt}"][data-real="0"]', offset=520, dur=0.8)
            await click(pg, f'#reviewcard button[data-id="{nxt}"][data-real="0"]', dur=0.7, settle=1.0)
        log("nos", ids=nos)
        await scroll_to(pg, "#answercard", dur=1.0)
        await mark(pg, "holds", ["#answercard", "#answercard .verify", ".checks"])
        await asyncio.sleep(2.0)
        await scroll_to(pg, "#answercard .audit", offset=330, dur=0.9)
        await click(pg, "#arun", dur=0.7, settle=1.6)
        await mark(pg, "robust", [".audit", "#aout", "#aout .verdict"])
        await asyncio.sleep(2.0)

        # ---- re-imaging and evidence
        await click(pg, "#reimgbtn", dur=0.8, settle=1.2)
        await scroll_to(pg, ".card.stage", offset=70, dur=1.0)
        await mark(pg, "reimage", ["#stagewrap"])
        await asyncio.sleep(2.5)
        await scroll_to(pg, "#answercard .tools", offset=520, dur=1.0)
        await move_to(pg, "#answercard .evid", dur=0.9)
        await mark(pg, "evidence", [".tools", ".reimage", "#answercard .evid"])
        await asyncio.sleep(2.0)

        # ---- the closing shot: the landing page
        await pg.goto(BASE + "/", wait_until="domcontentloaded")
        await restart()
        await asyncio.sleep(5.5)
        await mark(pg, "close", [".hero h1", ".specimen"])
        await asyncio.sleep(1.0)
        log("end")
        await cdp.send("Page.stopScreencast")
        await asyncio.sleep(0.5)
        await b.close()

    t0 = FRAME_LOG[0]["t"] if FRAME_LOG else EVENTS[0]["t"]
    for e in EVENTS:
        e["t"] = round(e["t"] - t0, 3)
    for f in FRAME_LOG:
        f["t"] = round(f["t"] - t0, 3)
    (HERE / "frames.json").write_text(json.dumps(FRAME_LOG))
    (HERE / "events.json").write_text(json.dumps(EVENTS, indent=1))
    print(f"{len(FRAME_LOG)} frames over {FRAME_LOG[-1]['t']:.1f}s; challenger {next((e['id'] for e in EVENTS if e['kind']=='challenger'), None)}; "
          f"nos {next((e['ids'] for e in EVENTS if e['kind']=='nos'), None)}")


asyncio.run(main())
