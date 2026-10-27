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

    def _zoomed(self, image: Image.Image, region: tuple[int, int, int, int], scale: float, fn) -> list[Instance]:
        """Run `fn(crop)` on an upscaled crop and map the masks back to full-image coordinates.

        SAM predicts masks on a coarse internal grid, so small cells get blobby edges;
        upscaling the crop gives each cell more mask pixels. Masks are downsampled with
        bilinear averaging, so edges land at sub-pixel accuracy.
        """
        l, t, r, b = region
        crop = image.crop(region)
        big = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
        out = []
        for inst in fn(big, scale):
            small = Image.fromarray(inst.mask.astype(np.uint8) * 255).resize(crop.size, Image.BILINEAR)
            full = np.zeros((image.height, image.width), dtype=bool)
            full[t:b, l:r] = np.asarray(small) >= 128
            if not full.any():
                continue
            ys, xs = np.nonzero(full)
            out.append(Instance(mask=full, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                                confidence=inst.confidence, prompt=inst.prompt))
        return out

    def segment_text_tiled(self, image: Image.Image, concept: str, tiles: int = 3, overlap: int = 32,
                           scale: float = 3.0) -> list[Instance]:
        """Text prompt on a tiles x tiles grid of upscaled, overlapping crops.

        Instances from overlapping tiles are de-duplicated by mask IoU (keep the more confident one).
        Instances touching a tile's inner edge are dropped; the neighbouring tile sees them whole.
        """
        w, h = image.size
        found: list[Instance] = []
        for row in range(tiles):
            for col in range(tiles):
                l, r = col * w // tiles, (col + 1) * w // tiles
                t, b = row * h // tiles, (row + 1) * h // tiles
                region = (max(0, l - overlap), max(0, t - overlap), min(w, r + overlap), min(h, b + overlap))
                for inst in self._zoomed(image, region, scale, lambda big, _s: self.segment_text(big, concept)):
                    bl, bt, br, bb = inst.box
                    on_inner_edge = ((bl <= region[0] and region[0] > 0) or (bt <= region[1] and region[1] > 0)
                                     or (br >= region[2] and region[2] < w) or (bb >= region[3] and region[3] < h))
                    if not on_inner_edge:
                        found.append(inst)
        return _dedupe(found)

    def segment_boxes_zoomed(self, image: Image.Image, boxes: Sequence[tuple[int, int, int, int]],
                             context: int = 48, scale: float = 3.0) -> list[Instance]:
        """Box prompts on upscaled windows: boxes are grouped into windows around them, 16 per request."""
        w, h = image.size
        out: list[Instance] = []
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] // 96, boxes[i][0]))
        for start in range(0, len(order), MAX_OBJECTS_PER_REQUEST):
            group = [boxes[i] for i in order[start:start + MAX_OBJECTS_PER_REQUEST]]
            region = (max(0, min(b[0] for b in group) - context), max(0, min(b[1] for b in group) - context),
                      min(w, max(b[2] for b in group) + context), min(h, max(b[3] for b in group) + context))
            # Keep the upscaled window near SAM's working size; very wide groups get less zoom.
            s = min(scale, 1008 / max(region[2] - region[0], region[3] - region[1]))
            s = max(s, 1.0)

            def run(big, s_, group=group, region=region):
                sx, sy = big.width / (region[2] - region[0]), big.height / (region[3] - region[1])
                scaled = [(int((l - region[0]) * sx), int((t - region[1]) * sy),
                           min(big.width, int(round((r - region[0]) * sx))), min(big.height, int(round((b - region[1]) * sy))))
                          for l, t, r, b in group]
                return self.segment_boxes(big, scaled)

            out.extend(self._zoomed(image, region, s, run))
        return out

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


def _dedupe(instances: list[Instance], thresh: float = 0.5) -> list[Instance]:
    """Greedy non-maximum suppression on mask IoU, most confident first."""
    keep: list[Instance] = []
    for inst in sorted(instances, key=lambda i: -(i.confidence or 0)):
        a = inst.mask.sum()
        if all(np.logical_and(inst.mask, k.mask).sum() / (a + k.mask.sum() - np.logical_and(inst.mask, k.mask).sum()) < thresh
               for k in keep if _boxes_overlap(inst.box, k.box)):
            keep.append(inst)
    return keep


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
