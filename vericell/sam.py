"""SAM 3.1 client: text or box prompts in, full-image instance masks + confidence out."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    FramePrompt,
    PromptBox,
    PromptObject,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    build_object_prompt,
    decode_mask_to_raster,
    image_segmentation_format,
)
from openai import OpenAI
from PIL import Image

MODEL = "sam-3.1"
MAX_OBJECTS_PER_REQUEST = 16


@dataclass
class Instance:
    mask: np.ndarray  # bool, image-sized
    box: tuple[int, int, int, int]  # left, top, right, bottom (half-open)
    confidence: float | None
    prompt: str  # what produced it, e.g. "text:cell" or "box"


def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _paste(record: SegmentationMaskRecord, width: int, height: int) -> np.ndarray:
    """Decode a mask record and place it into an image-sized boolean array.

    The mask raster covers the record's box; it may be coarser than the box, so it is
    resized (nearest) to fit before pasting.
    """
    m = record.mask
    raster = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    b = record.bounds
    left, top, right, bottom = int(b.left), int(b.top), int(b.right), int(b.bottom)
    if raster.shape != (bottom - top, right - left):
        raster = np.asarray(
            Image.fromarray(raster * 255).resize((right - left, bottom - top), Image.NEAREST)
        ) > 0
    full = np.zeros((height, width), dtype=bool)
    full[top:bottom, left:right] = raster.astype(bool)
    return full


class SAM:
    def __init__(self, client: OpenAI):
        self.client = client

    def _call(self, text: str, image: Image.Image) -> str:
        stream = self.client.responses.create(
            model=MODEL,
            input=[{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": to_data_url(image)},
                ],
            }],
            stream=True,
            metadata={"include_confidence": "true"},
        )
        return "".join(ev.delta for ev in stream if ev.type == "response.output_text.delta")

    def _parse(self, raw: str, image: Image.Image, prompt: str) -> list[Instance]:
        parser = image_segmentation_format().create_parser()
        parser.push(raw, emit=False)
        result = parser.finish(CompletedOutcome()).result
        boxes = {r.object_id: r for r in result.records if isinstance(r, SegmentationBoxRecord)}
        out = []
        for r in result.records:
            if not isinstance(r, SegmentationMaskRecord):
                continue
            box = boxes.get(r.object_id)
            conf = r.confidence if r.confidence is not None else (box.confidence if box else None)
            b = r.bounds
            out.append(Instance(
                mask=_paste(r, *image.size),
                box=(int(b.left), int(b.top), int(b.right), int(b.bottom)),
                confidence=conf,
                prompt=prompt,
            ))
        return out

    def segment_text(self, image: Image.Image, concept: str) -> list[Instance]:
        """All instances of one concept noun phrase (one concept per request)."""
        return self._parse(self._call(concept, image), image, f"text:{concept}")

    def segment_boxes(self, image: Image.Image, boxes: Sequence[tuple[int, int, int, int]]) -> list[Instance]:
        """One instance per box, batched 16 boxes per request. Boxes are half-open (l, t, r, b)."""
        out: list[Instance] = []
        for start in range(0, len(boxes), MAX_OBJECTS_PER_REQUEST):
            chunk = boxes[start:start + MAX_OBJECTS_PER_REQUEST]
            objects = [PromptObject(id=i, prompts=[FramePrompt(box=PromptBox(*map(int, b)))])
                       for i, b in enumerate(chunk)]
            text = build_object_prompt(size=image.size, objects=objects)
            out.extend(self._parse(self._call(text, image), image, "box"))
        return out
