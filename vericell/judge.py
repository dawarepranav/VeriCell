"""Crop judge: Muse Spark looks at one uncertain object and says what it is.

Used where the segmenters can't settle the question themselves, e.g. whether several touching
parts are one budding organoid or separate organoids that happen to touch.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from .sam import Instance, to_data_url

MODEL = "muse-spark-1.3"

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["one_object", "several_objects", "not_an_object"]},
        "confidence": {"type": "number", "description": "0 to 1"},
        "reason": {"type": "string", "description": "One short sentence a biologist would understand."},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You check microscopy segmentations. You see a crop of a brightfield image with one region
outlined in red. Decide what the outlined region contains:
- one_object: a single {target}
- several_objects: two or more separate {target}s that touch or sit close together
- not_an_object: debris, background, a bubble or out-of-focus blur, not a {target}

How {target}s look: a single {target} is often lobed, budding, branched or irregular. Buds, crypts,
lobes, several internal cavities (lumens), dark cores and uneven texture are all normal inside ONE
{target} and are NOT evidence of several. Call several_objects only if you can see a separation
between bodies: a thin line or gap of background, or two complete outer walls meeting with a
pinched seam between otherwise rounded bodies. When there is no such visible separation, it is
one_object. Judge only the outlined region; use confidence for how sure you are."""


def crop_with_outline(image: Image.Image, inst: Instance, pad: float = 0.35, side: int = 512) -> Image.Image:
    """Crop around the object (with context) and draw its outline in red."""
    arr = np.asarray(image.convert("RGB")).copy()
    l, t, r, b = inst.box
    p = int(max(r - l, b - t) * pad) + 8
    L, T, R, B = max(0, l - p), max(0, t - p), min(arr.shape[1], r + p), min(arr.shape[0], b + p)
    contours, _ = cv2.findContours(inst.mask[T:B, L:R].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crop = np.ascontiguousarray(arr[T:B, L:R])
    cv2.drawContours(crop, contours, -1, (255, 0, 0), max(1, int(max(crop.shape[:2]) / 250)))
    out = Image.fromarray(crop)
    out.thumbnail((side, side))
    return out


def judge(client: OpenAI, image: Image.Image, inst: Instance, target: str = "organoid") -> dict:
    r = client.responses.create(
        model=MODEL,
        instructions=SYSTEM.format(target=target),
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": f"Is the red-outlined region one {target}, several, or not a {target}?"},
            {"type": "input_image", "image_url": to_data_url(crop_with_outline(image, inst))},
        ]}],
        text={"format": {"type": "json_schema", "name": "crop_verdict", "schema": SCHEMA, "strict": True}},
    )
    out = json.loads(r.output_text)
    out["confidence"] = float(np.clip(out["confidence"], 0, 1))
    return out
