"""Render docs/video/cards.html to 1920x1080 PNGs in docs/video/cards (lower-thirds keep transparency).

    pip install playwright && playwright install chromium
    python docs/video/render_cards.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "cards"
NAMES = {"c01-hook": "01-hook-overlay", "c02-provenance": "02-provenance", "c04-truth": "04-truth",
         "c05a-benchmark": "05a-benchmark", "c05b-warning": "05b-warning", "c05c-honest": "05c-honest",
         "c05d-open": "05d-open-weights", "c07-close": "07-close", "lt-voice": "lt-voice",
         "lt-muse-spark": "lt-muse-spark", "lt-sam": "lt-sam", "lt-code": "lt-code", "lt-agent": "lt-agent"}

OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    try:
        browser = p.chromium.launch(channel="chrome")
    except Exception:
        browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.goto((HERE / "cards.html").as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    page.add_style_tag(content="body { background: transparent !important; }")
    for el_id, name in NAMES.items():
        page.locator(f"#{el_id}").screenshot(path=OUT / f"{name}.png", omit_background=True)
    browser.close()
print(sorted(f.name for f in OUT.glob("*.png")))
