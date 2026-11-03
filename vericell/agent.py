"""The VeriCell agent harness.

Design (after Anthropic's "Building effective agents" and 2026 bio-agent work such as Agentic-J and
the AI Neuroscientist): a fixed backbone where order matters, and model-directed control only
where judgement is needed.

  Backbone (deterministic order, one artifact per phase):
    look      Muse Spark describes the image and names the objects in SAM's vocabulary
    plan      Muse Spark turns the question into a strict JSON plan
    segment   Cellpose-SAM (+ coarser pyramid levels, merged lobes) and SAM 3.1 text prompts
    verify    calibrated per-object P(real) from the verifier
  Agent loop (Muse Spark chooses tools, the harness enforces the rules):
    judge_objects    Spark looks at crops of chosen objects: one object / several / not an object
    try_sam_prompt   re-run SAM 3.1 with a different phrase (e.g. when it doesn't know "organoid")
    ask_human        hand specific objects to the user's review queue, then stop
    finish           stop: the answer is good enough
  Answer: deterministic geometry + a calibrated 90% range (answers.py); no model touches a number.

Harness guarantees:
  - policy gate on every tool call: known tool, schema-checked arguments, objects that exist,
    per-tool and total budgets, no repeated identical calls
  - evaluator after every action: the answer and its range are recomputed and the change is logged;
    the loop stops when the range is tight, nothing uncertain could move the answer, the controller
    finishes, two actions in a row don't help, or a budget runs out
  - deterministic fallback policy whenever the controller fails or proposes an invalid call
  - a full trace (every phase, tool call, argument, effect, timing) streamed live and saved to
    outputs/runs/<run_id>.json
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import answers, api, llm, pipeline, proposers
from .features import Candidate, add_judgement, reasons
from .judge import judge
from .measure import measure_masks
from .planner import Planner, describe_image
from .sam import SAM, _dedupe
from .verifier import matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "outputs" / "models"
RUNS = ROOT / "outputs" / "runs"
REVIEW_BAND = (0.2, 0.8)
TARGET_REL_WIDTH = 0.10      # stop once the 90% range is within +-10% of the answer
MIN_IMPACT = 0.02            # objects that can't move the answer by 2% aren't worth an action

METRIC_LABEL = {"Area_outer": ("area", "px²"), "Perimeter_outer": ("perimeter", "px"),
                "roundness": ("roundness", ""), "perimeter_diff": ("roughness (boundary excess)", "px"),
                "count": ("count", "")}


@dataclass
class Budget:
    max_steps: int = 6           # controller decisions in the agent loop
    max_judged: int = 18         # crop-judge calls in total
    max_prompts: int = 2         # extra SAM phrasings
    max_seconds: float = 240.0
    used_steps: int = 0
    used_judged: int = 0
    used_prompts: int = 0

    def left(self) -> dict:
        return {"steps": self.max_steps - self.used_steps, "judge_calls": self.max_judged - self.used_judged,
                "sam_prompts": self.max_prompts - self.used_prompts}


@dataclass
class Session:
    id: str
    image: Image.Image
    question: str
    name: str | None
    cache: Path | None
    scene: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    domain: str = "cell"
    props: pipeline.Proposals | None = None
    cands: list[Candidate] = field(default_factory=list)
    p: np.ndarray = field(default_factory=lambda: np.zeros(0))
    table: list = field(default_factory=list)
    verdicts: dict = field(default_factory=dict)        # box key -> crop-judge verdict (survives rebuilds)
    overrides: dict[int, bool] = field(default_factory=dict)
    agent_flagged: list[int] = field(default_factory=list)
    tried_phrases: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    trace: list[dict] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def probs(self) -> np.ndarray:
        p = self.p.copy()
        for i, real in self.overrides.items():
            p[i] = 1.0 if real else 0.0
        return p


# ---------------------------------------------------------------- tool schemas (what the controller sees)
TOOLS = [
    {"type": "function", "name": "judge_objects", "strict": True,
     "description": "Show Muse Spark a crop of each object and ask: one object, several touching objects, "
                    "or not an object (debris/blur). Use on uncertain objects that could move the answer, "
                    "especially merged/lobed ones or ones only one model found. Max 6 ids per call.",
     "parameters": {"type": "object", "properties": {
         "object_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 6},
         "why": {"type": "string", "description": "One sentence: why these objects."}},
         "required": ["object_ids", "why"], "additionalProperties": False}},
    {"type": "function", "name": "try_sam_prompt", "strict": True,
     "description": "Run the SAM 3.1 segmenter again with a different short visual noun phrase. Use when SAM "
                    "found far fewer objects than Cellpose (it may not know the biology word).",
     "parameters": {"type": "object", "properties": {
         "phrase": {"type": "string", "description": "2-4 plain visual words, e.g. 'round cell cluster'."},
         "why": {"type": "string"}}, "required": ["phrase", "why"], "additionalProperties": False}},
    {"type": "function", "name": "ask_human", "strict": True,
     "description": "Hand specific objects to the human's review queue (they will mark them real / not real). "
                    "Use when the remaining doubt can't be resolved automatically. Ends the loop.",
     "parameters": {"type": "object", "properties": {
         "object_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 5},
         "message": {"type": "string", "description": "One sentence to the human."}},
         "required": ["object_ids", "message"], "additionalProperties": False}},
    {"type": "function", "name": "finish", "strict": True,
     "description": "Stop: the answer is good enough or nothing useful is left to do.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                    "required": ["reason"], "additionalProperties": False}},
]

CONTROLLER_PROMPT = """You are the controller of a microscopy analysis agent. Deterministic code has already
segmented the image, scored every object's probability of being real, and computed an answer with a
calibrated 90% range. Your job: choose the ONE next action that most reduces the answer's uncertainty,
or finish. You never compute numbers yourself.

Guidance:
- Prefer judge_objects on the uncertain objects with the highest impact on the answer.
- If SAM found far fewer objects than Cellpose, its prompt may not match the biology word: try_sam_prompt
  with a plain visual phrase (avoid phrases already tried).
- If the top remaining objects are ones a crop can't settle, ask_human with those ids.
- finish when the range is already tight, nothing uncertain has meaningful impact, or actions stopped helping.
Call exactly one tool."""


def _load(domain: str):
    model = pickle.loads((MODELS / f"{domain}.pkl").read_bytes())["model"]
    cpath = MODELS / f"{domain}_calibration.json"
    calib = json.loads(cpath.read_text())["q"] if cpath.exists() else None
    return model, calib


class Agent:
    def __init__(self):
        self.client = api.client()
        self.sam = SAM(self.client)
        self.planner = Planner(self.client)
        self.models = {d: _load(d) for d in ("organoid", "cell") if (MODELS / f"{d}.pkl").exists()}
        self.sessions: dict[str, Session] = {}

    # ================================================================ run
    def run(self, image: Image.Image, question: str, name: str | None = None, cache: Path | None = None,
            llm_backend: str | None = None):
        llm.use(llm_backend)  # "meta" (hosted Muse Spark) or "open" (Muse Spark Open on Fireworks)
        s = Session(uuid.uuid4().hex[:12], image.convert("RGB"), question, name, cache)
        self.sessions[s.id] = s

        def emit(step, title, detail="", **data):
            ev = {"step": step, "title": title, "detail": detail, "data": data, "t": round(time.time() - s.started, 1)}
            s.trace.append({"step": step, "title": title, "detail": detail, "t": ev["t"], "data": _brief(data)})
            return ev

        img = s.image
        yield emit("start", "Reading the image", f"{img.width}×{img.height} px · reasoning model: {llm.MODELS[llm.backend()]}",
                   run=s.id, llm=llm.backend())

        # -- look
        suffix = "scene.json" if llm.backend() == "meta" else f"scene_{llm.backend()}.json"
        sp = cache / f"{name}.{suffix}" if cache and name else None
        s.scene = json.loads(sp.read_text()) if sp and sp.exists() else describe_image(self.client, img)
        if sp and not sp.exists():
            sp.write_text(json.dumps(s.scene))
        s.tried_phrases = list(s.scene["sam_concepts"][:2])
        yield emit("look", f"Muse Spark sees: {s.scene['target_object']}s ({s.scene['modality']})",
                   s.scene["notes"], concepts=s.scene["sam_concepts"])

        # -- plan
        s.plan = self.planner.plan(question)
        label, _ = METRIC_LABEL.get(s.plan["metric"], (s.plan["metric"], ""))
        what = "count the objects" if s.plan["operation"] == "count" else f"{s.plan['operation'].replace('_', ' ')} of {label}"
        yield emit("plan", f"Plan: {what}", s.plan["interpretation"], plan=s.plan)

        # -- segment (each proposer reported as it finishes)
        path = (lambda suffix: cache / f"{name}.{suffix}") if cache and name else (lambda suffix: None)
        cp = pipeline._cached(path("cellpose.json"), lambda: proposers.cellpose(img))
        coarse = []
        for sc in proposers.pyramid_scales(img)[1:]:
            coarse += pipeline._cached(path(f"cellpose_s{sc:.3f}.json"), lambda sc=sc: proposers.cellpose_scaled(img, sc))
        merged = proposers.touching_clusters(cp)
        yield emit("segment", f"Cellpose-SAM found {len(cp)} objects",
                   f"plus {len(coarse)} at coarser scales and {len(merged)} groups of touching parts")
        tiles, scale = pipeline.tiling(img)

        def text():
            found = []
            for concept in s.tried_phrases:
                found += self.sam.segment_text_tiled(img, concept, tiles=tiles, scale=scale)
            return _dedupe(found)
        tx = pipeline._cached(path("sam_text.json"), text)
        zm = pipeline._cached(path("zoom.json"), lambda: self.sam.segment_boxes_zoomed(img, [i.box for i in cp]))
        yield emit("segment", f"SAM 3.1 found {len(tx)} objects",
                   f"prompts: {', '.join(s.tried_phrases)}; plus a zoomed second opinion on every Cellpose object")
        s.props = pipeline.Proposals(s.scene, cp, coarse, merged, tx, zm)

        # -- verify
        s.domain = "organoid" if "organoid" in s.scene["target_object"].lower() and "organoid" in self.models else "cell"
        self._rebuild(s)
        p = s.p
        yield emit("verify", f"Scored all {len(s.cands)} candidate objects",
                   f"{int((p >= 0.5).sum())} look real, {int(((p >= REVIEW_BAND[0]) & (p < REVIEW_BAND[1])).sum())} uncertain",
                   hist=np.histogram(p, bins=10, range=(0, 1))[0].tolist())

        result = self.answer(s)
        yield emit("draft", f"First answer: {result['text']}", f"range ±{result['rel_width']:.0%} of the answer",
                   result=result)

        # -- agent loop: controller proposes, gate checks, tool runs, evaluator measures the effect
        no_gain = 0
        while True:
            stop = self._should_stop(s, result, no_gain)
            if stop:
                yield emit("stop", "Done checking", stop)
                break
            s.budget.used_steps += 1
            call, source = self._decide(s, result)
            ok, why_not = self._gate(s, call)
            if not ok:
                yield emit("gate", f"Policy gate rejected {call['name']}", why_not)
                call, source = self._fallback(s, result), "fallback"
                ok, why_not = self._gate(s, call)
                if not ok:
                    yield emit("stop", "Done checking", "no valid action left")
                    break
            if call["name"] == "finish":
                yield emit("stop", "Agent decided to finish" + ("" if source == "controller" else " (fallback policy)"),
                           call["args"].get("reason", ""))
                break
            before = result
            effect = yield from self._execute(s, call, emit, source)
            result = self.answer(s)
            gain = before["rel_width"] - result["rel_width"]
            no_gain = no_gain + 1 if gain <= 0.01 else 0
            yield emit("effect", f"Answer now: {result['text']}",
                       f"range ±{before['rel_width']:.0%} → ±{result['rel_width']:.0%}" + (f" · {effect}" if effect else ""),
                       result=result)
            if call["name"] == "ask_human":
                break

        result = self.answer(s)
        yield emit("answer", result["text"], f"{len(s.trace)} trace events · run {s.id}", result=result, session=s.id)
        self._save(s)

    # ================================================================ loop machinery
    def _rebuild(self, s: Session) -> None:
        """(Re)build candidates from the current proposals, re-apply judge verdicts, re-score."""
        s.cands = pipeline.candidates(s.image, s.props)
        for c in s.cands:
            v = s.verdicts.get(str(list(c.inst.box)))
            if v is not None:
                add_judgement(c, v)
        model, _ = self.models[s.domain]
        s.p = model.predict_proba(matrix(s.cands))[:, 1]
        s.table = [(measure_masks([c.inst.mask]) or [None])[0] for c in s.cands]

    def _should_stop(self, s: Session, result: dict, no_gain: int) -> str:
        b = s.budget
        if b.used_steps >= b.max_steps:
            return "step budget used up"
        if time.time() - s.started > b.max_seconds:
            return "time budget used up"
        if no_gain >= 2:
            return "the last two actions didn't narrow the range"
        if result["rel_width"] <= TARGET_REL_WIDTH:
            return f"the range is already within ±{TARGET_REL_WIDTH:.0%}"
        if not any(q["impact"] >= MIN_IMPACT for q in result["review"]) and not self._sam_looks_lost(s):
            return "no uncertain object could move the answer"
        return ""

    def _sam_looks_lost(self, s: Session) -> bool:
        return len(s.props.sam) < 0.3 * max(len(s.props.cellpose), 1) and s.budget.used_prompts < s.budget.max_prompts

    def _summary(self, s: Session, result: dict) -> str:
        """Compact state for the controller: never raw masks, only what a decision needs."""
        items = [{"id": q["id"], "p_real": q["p"], "impact_on_answer": q["impact"], "kind": s.cands[q["id"]].source,
                  "judged": str(list(s.cands[q["id"]].inst.box)) in s.verdicts, "warnings": q["reasons"][:2]}
                 for q in result["review"][:8]]
        return json.dumps({
            "question": s.question, "target": s.scene["target_object"],
            "plan": {k: s.plan[k] for k in ("metric", "operation")},
            "answer": result["text"], "range_relative_halfwidth": round(result["rel_width"], 3),
            "objects_found": {"cellpose": len(s.props.cellpose), "sam": len(s.props.sam)},
            "sam_phrases_tried": s.tried_phrases, "uncertain_objects_by_impact": items,
            "budget_left": s.budget.left(), "actions_so_far": [t["title"] for t in s.trace if t["step"] == "tool"],
        })

    def _decide(self, s: Session, result: dict) -> tuple[dict, str]:
        try:
            r = llm.complete(CONTROLLER_PROMPT, self._summary(s, result), tools=TOOLS)
            if r.tool is None:
                raise RuntimeError("controller made no tool call")
            return {"name": r.tool, "args": r.args}, "controller"
        except Exception as exc:  # controller unavailable or no tool call: the gate rejects, fallback runs
            return {"name": "_error", "args": {"error": str(exc)[:200]}}, "controller"

    def _fallback(self, s: Session, result: dict) -> dict:
        """Deterministic policy: rephrase SAM if it looks lost, else judge the top unjudged objects, else finish."""
        if self._sam_looks_lost(s):
            for phrase in ("round cell cluster", "cell cluster", "cell", "blob"):
                if phrase not in (p.lower() for p in s.tried_phrases):
                    return {"name": "try_sam_prompt", "args": {"phrase": phrase, "why": "SAM found few objects"}}
        todo = [q["id"] for q in result["review"]
                if q["impact"] >= MIN_IMPACT and str(list(s.cands[q["id"]].inst.box)) not in s.verdicts][:4]
        if todo and s.budget.used_judged + len(todo) <= s.budget.max_judged:
            return {"name": "judge_objects", "args": {"object_ids": todo, "why": "highest-impact unjudged objects"}}
        return {"name": "finish", "args": {"reason": "nothing useful left to try"}}

    def _gate(self, s: Session, call: dict) -> tuple[bool, str]:
        name, args = call["name"], call["args"]
        if name not in {t["name"] for t in TOOLS}:
            return False, f"unknown tool ({args.get('error', name)})"
        b = s.budget
        ids = args.get("object_ids", [])
        if any(not isinstance(i, int) or not 0 <= i < len(s.cands) for i in ids):
            return False, "refers to objects that don't exist"
        if name == "judge_objects":
            fresh = [i for i in ids if str(list(s.cands[i].inst.box)) not in s.verdicts]
            if not fresh:
                return False, "all of those objects were already judged"
            if b.used_judged + len(fresh) > b.max_judged:
                return False, "crop-judge budget would be exceeded"
        if name == "try_sam_prompt":
            phrase = args.get("phrase", "").strip().lower()
            if not phrase or len(phrase.split()) > 5:
                return False, "phrase must be 1-5 plain words"
            if phrase in (p.lower() for p in s.tried_phrases):
                return False, f"'{phrase}' was already tried"
            if b.used_prompts >= b.max_prompts:
                return False, "SAM re-prompt budget used up"
        return True, ""

    def _execute(self, s: Session, call: dict, emit, source: str):
        name, args = call["name"], call["args"]
        tag = "" if source == "controller" else " (fallback policy)"
        if name == "judge_objects":
            ids = [i for i in args["object_ids"] if str(list(s.cands[i].inst.box)) not in s.verdicts]
            yield emit("tool", f"judge_objects {ids}{tag}", args.get("why", ""), tool=name, args=args)
            before = {i: float(s.p[i]) for i in ids}
            boxes = {i: str(list(s.cands[i].inst.box)) for i in ids}
            with ThreadPoolExecutor(6) as ex:
                which = llm.backend()  # worker threads do not inherit the per-run choice; pass it explicitly
                verdicts = list(ex.map(lambda i: judge(None, s.image, s.cands[i].inst, s.scene["target_object"], which=which), ids))
            s.budget.used_judged += len(ids)
            for i, v in zip(ids, verdicts):
                s.verdicts[boxes[i]] = v
            self._rebuild(s)  # same proposals -> same object ids, so overrides stay valid
            lines = [f"#{i}: {v['verdict'].replace('_', ' ')} ({v['confidence']:.0%}), P {before[i]:.2f}→{s.p[i]:.2f}"
                     for i, v in zip(ids, verdicts)]
            yield emit("tool_result", "Muse Spark's verdicts", " · ".join(lines),
                       repairs=[{"id": i, "verdict": v["verdict"], "reason": v["reason"], "p_before": round(before[i], 2),
                                 "p_after": round(float(s.p[i]), 2)} for i, v in zip(ids, verdicts)])
            return f"{len(ids)} objects judged"
        if name == "try_sam_prompt":
            phrase = args["phrase"].strip()
            yield emit("tool", f"try_sam_prompt “{phrase}”{tag}", args.get("why", ""), tool=name, args=args)
            tiles, scale = pipeline.tiling(s.image)
            new = self.sam.segment_text_tiled(s.image, phrase, tiles=tiles, scale=scale)
            s.budget.used_prompts += 1
            s.tried_phrases.append(phrase)
            n_before = len(s.props.sam)
            s.props.sam = _dedupe(s.props.sam + new)
            s.overrides, s.agent_flagged = {}, []  # object ids change when proposals change
            self._rebuild(s)
            yield emit("tool_result", f"SAM 3.1 found {len(new)} objects with “{phrase}”",
                       f"{len(s.props.sam) - n_before} new after removing duplicates")
            return f"SAM objects {n_before} → {len(s.props.sam)}"
        if name == "ask_human":
            s.agent_flagged = [int(i) for i in args["object_ids"]]
            yield emit("tool", f"ask_human {s.agent_flagged}{tag}", args.get("message", ""), tool=name, args=args)
            return "handed to you for review"
        return ""

    # ================================================================ answers
    def answer(self, s: Session) -> dict:
        p = s.probs()
        _, calib = self.models[s.domain]
        idx = answers.pool(s.cands, p, s.table)
        value, winner = answers.point(s.plan, s.table, p, idx)
        hw = answers.mc_halfwidth(s.plan, s.table, p, idx) if value else {}
        rng = answers.ranges(s.plan, value, hw, calib)
        main = next((k for k in (value or {}) if k not in ("x", "y") and value[k] is not None), None)
        rel = (rng[main][1] - rng[main][0]) / 2 / max(abs(value[main]), 1e-9) if main in rng else 1.0
        return {"value": value, "range90": rng, "rel_width": float(rel), "winner": winner,
                "text": self._text(s, value, rng), "objects": self._objects(s, p, idx, winner),
                "review": self._review_queue(s, p, idx, value), "n_reviewed": len(s.overrides),
                "domain": s.domain, "plan": s.plan, "calibrated": calib is not None}

    def _review_queue(self, s: Session, p: np.ndarray, idx: list[int], value) -> list[dict]:
        """Uncertain objects ranked by how much the answer would move if resolved; agent-flagged first."""
        out = []
        for i in idx:
            if i in s.overrides or not (REVIEW_BAND[0] <= p[i] < REVIEW_BAND[1] or i in s.agent_flagged):
                continue
            moved = []
            for v in (1.0, 0.0):
                q = p.copy()
                q[i] = v
                moved.append(answers.point(s.plan, s.table, q, idx)[0])
            impact = 0.0
            for k, base in (value or {}).items():
                if isinstance(base, (int, float)) and k not in ("x", "y") and moved[0] and moved[1] \
                        and moved[0].get(k) is not None and moved[1].get(k) is not None:
                    impact = max(impact, abs(moved[0][k] - moved[1][k]) / max(abs(base), 1e-9))
            v = s.verdicts.get(str(list(s.cands[i].inst.box)))
            out.append({"id": i, "p": round(float(p[i]), 3), "impact": round(impact, 4), "box": list(s.cands[i].inst.box),
                        "reasons": reasons(s.cands[i].features), "agent_flagged": i in s.agent_flagged,
                        "judge": v["verdict"] if v else None})
        return sorted(out, key=lambda r: (not r["agent_flagged"], -r["impact"], abs(r["p"] - 0.5)))[:12]

    def _objects(self, s: Session, p: np.ndarray, idx: list[int], winner) -> list[dict]:
        import cv2
        out = []
        for i in idx:
            m = s.cands[i].inst.mask
            l, t, r, b = s.cands[i].inst.box
            cs, _ = cv2.findContours(m[t:b, l:r].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            poly = (cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.8, True)[:, 0, :] + [l, t]).tolist()
            out.append({"id": i, "p": round(float(p[i]), 3), "poly": poly, "box": [l, t, r, b], "source": s.cands[i].source,
                        "winner": i == winner, "reviewed": i in s.overrides, "flagged": i in s.agent_flagged,
                        "reasons": reasons(s.cands[i].features)})
        return out

    def _text(self, s: Session, value, rng) -> str:
        if not value:
            return f"I couldn't find any {s.scene['target_object']}s to measure."
        plan = s.plan
        label, unit = METRIC_LABEL.get(plan["metric"], (plan["metric"], ""))
        u = f" {unit}" if unit else ""
        fmt = (lambda v: f"{v:.3f}") if plan["metric"] == "roundness" else (lambda v: f"{v:,.0f}")
        r = lambda k: f" (90% range {fmt(rng[k][0])}–{fmt(rng[k][1])})" if k in rng else ""  # noqa: E731
        target = s.scene["target_object"]
        if plan["operation"] in ("max", "min"):
            big = plan["metric"] in ("Area_outer", "Perimeter_outer")
            word = ("largest" if big else "highest") if plan["operation"] == "max" else ("smallest" if big else "lowest")
            out = f"The {target} with the {word} {label}: {fmt(value[plan['metric']])}{u}{r(plan['metric'])}"
            if "x" in value:
                out += f", centred at ({value['x']:.0f}, {value['y']:.0f})"
            return out + "."
        if plan["operation"] == "tertile_mean":
            names = plan["tier_names"] if len(plan["tier_names"]) == 3 else ["low", "mid", "high"]
            return f"Tier means of {label}: " + "; ".join(
                f"{n} {fmt(value[k])}{u}{r(k)}" for n, k in zip(names, ("low", "mid", "high")) if value[k] is not None) + "."
        if plan["operation"] == "count":
            return f"About {value['count']:.0f} {target}s{r('count')}."
        return f"{plan['operation'].title()} {label}: {fmt(value[plan['metric']])}{u}{r(plan['metric'])}."

    # ================================================================ human review + persistence
    def review(self, session_id: str, object_id: int, real: bool) -> dict:
        s = self.sessions[session_id]
        s.overrides[int(object_id)] = bool(real)
        s.trace.append({"step": "human", "title": f"You marked #{object_id} as {'real' if real else 'not real'}",
                        "detail": "", "t": round(time.time() - s.started, 1), "data": {}})
        self._save(s)
        return self.answer(s)

    def _save(self, s: Session) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{s.id}.json").write_text(json.dumps({
            "run": s.id, "question": s.question, "image": s.name or "upload", "scene": s.scene, "plan": s.plan,
            "domain": s.domain, "budget": vars(s.budget), "verdicts": s.verdicts, "trace": s.trace}, indent=1, default=str))


def _brief(data: dict) -> dict:
    """Trace copy without bulky fields (outlines are re-derivable from the run)."""
    out = {}
    for k, v in data.items():
        out[k] = {x: v[x] for x in ("text", "value", "range90", "rel_width", "winner")} if k == "result" else v
    return out
