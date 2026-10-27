"""Planner: Muse Spark turns a biologist's question into a strict, executable JSON plan.

The plan says what to segment (with noun phrases SAM 3.1 understands), which deterministic metric
to compute, and which operation to run over the population. It never computes numbers itself.
"""

from __future__ import annotations

import json

from openai import OpenAI
from PIL import Image

from .sam import to_data_url

MODEL = "muse-spark-1.3"

METRICS = ["Area_outer", "Perimeter_outer", "roundness", "perimeter_diff", "count"]
OPERATIONS = ["max", "min", "tertile_mean", "mean", "median", "count"]
FIELDS = ["Area_outer", "Perimeter_outer", "x", "y", "roundness", "perimeter_diff", "count"]

SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string", "description": "The question restated in precise, measurable terms."},
        "target_object": {"type": "string", "description": "The biological object to segment, e.g. 'organoid', 'cell', 'nucleus'."},
        "sam_concepts": {
            "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
            "description": "1-3 short, concrete visual noun phrases for a segmentation model, most likely first.",
        },
        "metric": {"type": "string", "enum": METRICS},
        "operation": {"type": "string", "enum": OPERATIONS},
        "fields": {"type": "array", "items": {"type": "string", "enum": FIELDS}, "minItems": 1},
        "tier_names": {"type": "array", "items": {"type": "string"},
                       "description": "For tertile_mean: the user's names for the low/mid/high tiers, else empty."},
    },
    "required": ["interpretation", "target_object", "sam_concepts", "metric", "operation", "fields", "tier_names"],
    "additionalProperties": False,
}

SYSTEM = """You plan quantitative analyses of microscopy images. You do not measure anything yourself:
deterministic code runs your plan. Map the user's wording onto exactly one metric and one operation.

Metrics (per object, from its outer contour, in pixels):
- Area_outer: area / size / footprint / cross-sectional surface / how big / how much space it occupies.
- Perimeter_outer: perimeter / boundary or edge length / outline length / circumference.
- roundness: 4*pi*A/P^2, a 0-1 circularity index. Words: round, circular, spherical, symmetrical,
  regular shape; "irregular" means LOW roundness.
- perimeter_diff: P - 2*sqrt(pi*A), the boundary excess over a circle of the same area. Words:
  roughness, rugosity, folded / budding / crenulated margins, deviation from a smooth outline,
  perimeter difference, boundary excess.
- count: number of objects.

Operations:
- max / min: the single most / least extreme object. Words like "largest", "most dominant",
  "longest", "most spherical", "most irregular boundary" mean max of the matching metric.
- tertile_mean: split the population into three equal-count tiers by the metric (low/mid/high,
  small/medium/large, spherical/intermediate/irregular...) and report each tier's mean.
- mean / median / count: population summaries.

Fields to report: for max/min, the metric itself, plus x and y when the user asks where the object
is (center, centroid, location, coordinates, "center of mass"). For tertile_mean, the metric.

sam_concepts are for a segmentation model that only understands concrete visual nouns. Use plain
phrases such as "organoid", "cell", "nucleus", "round cell cluster"; never metric words."""


class Planner:
    def __init__(self, client: OpenAI):
        self.client = client
        self._cache: dict[str, dict] = {}

    def plan(self, question: str) -> dict:
        if question not in self._cache:
            r = self.client.responses.create(
                model=MODEL,
                instructions=SYSTEM,
                input=question,
                text={"format": {"type": "json_schema", "name": "analysis_plan", "schema": SCHEMA, "strict": True}},
            )
            plan = json.loads(r.output_text)
            # Deterministic guard-rails: the operation decides which fields are meaningful.
            if plan["operation"] == "tertile_mean":
                plan["fields"] = [plan["metric"]]
            elif plan["operation"] in ("max", "min") and plan["metric"] not in plan["fields"]:
                plan["fields"].insert(0, plan["metric"])
            self._cache[question] = plan
        return self._cache[question]


SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "target_object": {"type": "string", "description": "The main countable biological objects, e.g. 'organoid'."},
        "sam_concepts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
                         "description": "1-3 short visual noun phrases a segmentation model would understand, best first."},
        "modality": {"type": "string", "description": "e.g. brightfield, phase contrast, fluorescence."},
        "notes": {"type": "string", "description": "One sentence on what makes this image hard to segment, if anything."},
    },
    "required": ["target_object", "sam_concepts", "modality", "notes"],
    "additionalProperties": False,
}

SCENE_SYSTEM = """You look at a microscopy image and name the main countable objects in it, so a
text-prompted segmentation model (SAM 3.1) can find them. Give plain visual noun phrases
("organoid", "cell", "round cell cluster", "nucleus"), most likely first. No metric words."""


def describe_image(client: OpenAI, image: Image.Image, max_side: int = 1024) -> dict:
    """One look at the image: which objects to segment, phrased for SAM. Grounds the planner."""
    small = image.copy()
    small.thumbnail((max_side, max_side))
    r = client.responses.create(
        model=MODEL,
        instructions=SCENE_SYSTEM,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "What objects should be segmented in this image?"},
            {"type": "input_image", "image_url": to_data_url(small.convert("RGB"))},
        ]}],
        text={"format": {"type": "json_schema", "name": "scene", "schema": SCENE_SCHEMA, "strict": True}},
    )
    return json.loads(r.output_text)
