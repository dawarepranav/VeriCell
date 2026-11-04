"""Render the README diagrams (docs/diagrams/diagrams.html) to light and dark PNGs in docs/images.

    pip install playwright && playwright install chromium
    python scripts/render_diagrams.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "docs" / "diagrams" / "diagrams.html").as_uri()
OUT = ROOT / "docs" / "images"

with sync_playwright() as p:
    try:
        browser = p.chromium.launch(channel="chrome")  # an installed Chrome, if there is one
    except Exception:
        browser = p.chromium.launch()
    for scheme in ("light", "dark"):
        page = browser.new_page(device_scale_factor=2, color_scheme=scheme, viewport={"width": 1240, "height": 1400})
        page.goto(SRC, wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.reload(wait_until="networkidle")  # lay text out again now the fonts are cached
        page.evaluate("document.fonts.ready")
        for name in ("arch", "loop", "verify"):
            page.locator(f"#{name}").screenshot(path=OUT / f"diagram-{name}-{scheme}.png")
        page.close()
    browser.close()
print("rendered", sorted(f.name for f in OUT.glob("diagram-*.png")))
