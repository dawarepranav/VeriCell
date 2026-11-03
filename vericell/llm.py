"""One interface to the language model, two backends.

  meta  Muse Spark 1.3 on the Meta Model API (hosted)
  open  Muse Spark Open on Fireworks (open weights)

Choose with VERICELL_LLM=meta|open (default meta). Every reasoning step (scene description,
planner, crop judge, agent controller) goes through `complete`, so the whole reasoning stack can run
on open weights with no other change. SAM 3.1 segmentation always uses the Meta API.
"""

from __future__ import annotations

import contextvars
import json
import os
from dataclasses import dataclass

from openai import OpenAI

from .api import BASE_URL, load_key

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
MODELS = {"meta": "muse-spark-1.3", "open": "accounts/fireworks/models/muse-spark-open"}


@dataclass
class Reply:
    text: str = ""
    tool: str | None = None
    args: dict | None = None
    usage: dict | None = None


_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("vericell_llm", default=None)


def backend() -> str:
    b = (_override.get() or os.environ.get("VERICELL_LLM", "meta")).lower()
    return b if b in MODELS else "meta"


def use(which: str | None) -> None:
    """Pick the backend for the current context (e.g. one agent run); None falls back to VERICELL_LLM."""
    _override.set(which if which in MODELS else None)


_clients: dict[str, OpenAI] = {}


def client(which: str | None = None) -> OpenAI:
    which = which or backend()
    if which not in _clients:
        load_key()  # also loads .env into the environment
        if which == "open":
            key = os.environ.get("FIREWORKS_API_KEY")
            if not key:
                raise RuntimeError("VERICELL_LLM=open needs FIREWORKS_API_KEY in .env")
            _clients[which] = OpenAI(base_url=FIREWORKS_URL, api_key=key)
        else:
            _clients[which] = OpenAI(base_url=BASE_URL, api_key=load_key())
    return _clients[which]


def _to_chat_content(parts: list[dict] | str):
    """Responses-style parts (input_text / input_image) -> chat-completions parts."""
    if isinstance(parts, str):
        return parts
    out = []
    for p in parts:
        if p["type"] == "input_text":
            out.append({"type": "text", "text": p["text"]})
        elif p["type"] == "input_image":
            out.append({"type": "image_url", "image_url": {"url": p["image_url"]}})
    return out


def complete(instructions: str, content: list[dict] | str, schema: dict | None = None, name: str = "out",
             tools: list[dict] | None = None, which: str | None = None, effort: str | None = None) -> Reply:
    """One model call. `content` uses Responses-API parts; `schema` forces strict JSON; `tools` asks for a call.

    `effort` (open backend only): "low" for simple structured tasks; None keeps the model's full
    reasoning. On our checks, low effort kept the planner at 16/16 and was ~7x faster, but dropped
    the crop judge from 70% to 60% balanced accuracy, so the judge and controller use full reasoning.
    """
    which = which or backend()
    c = client(which)
    if which == "meta":
        kwargs = {}
        if schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}
        if tools:
            kwargs["tools"] = tools
        user = content if isinstance(content, str) else [{"type": "message", "role": "user", "content": content}]
        r = c.responses.create(model=MODELS[which], instructions=instructions, input=user, **kwargs)
        call = next((i for i in r.output if i.type == "function_call"), None)
        if call is not None:
            return Reply(tool=call.name, args=json.loads(call.arguments))
        return Reply(text=r.output_text)

    # Fireworks: chat completions (OpenAI-compatible)
    kwargs = {}
    if schema:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
    if tools:
        kwargs["tools"] = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
                           for t in tools]
        kwargs["tool_choice"] = "required"
    if effort in ("low", "medium", "high"):
        kwargs["reasoning_effort"] = effort
    r = c.chat.completions.create(
        model=MODELS[which], max_tokens=4096,
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": _to_chat_content(content)}],
        **kwargs)
    msg = r.choices[0].message
    usage = r.usage.model_dump() if r.usage else None
    if msg.tool_calls:
        call = msg.tool_calls[0].function
        return Reply(tool=call.name, args=json.loads(call.arguments), usage=usage)
    return Reply(text=msg.content or "", usage=usage)


def _first_json(text: str) -> dict:
    """The first complete JSON object in `text` (tolerates code fences and trailing run-on text)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def complete_json(instructions: str, content, schema: dict, name: str = "out", which: str | None = None,
                  retries: int = 2, effort: str | None = None) -> dict:
    """Strict-JSON call; open models occasionally run on past the object, so parse leniently and retry."""
    last = None
    for _ in range(retries + 1):
        try:
            obj = _first_json(complete(instructions, content, schema=schema, name=name, which=which, effort=effort).text)
            missing = [k for k in schema.get("required", []) if k not in obj]
            if missing:
                raise ValueError(f"missing fields {missing}")
            return obj
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
    raise RuntimeError(f"no valid JSON after {retries + 1} tries: {last}")
