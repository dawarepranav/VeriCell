"""Muse Voice Transcribe: spoken question -> text."""

import json

import httpx

from .api import BASE_URL, load_key

MODEL = "muse-voice-transcribe-1.0"


def transcribe(wav: bytes) -> str:
    """`wav` must be 16-bit mono PCM WAV at 16 or 24 kHz (the browser UI records exactly that)."""
    resp = httpx.post(
        f"{BASE_URL}/asr/transcribe",
        headers={"Authorization": f"Bearer {load_key()}"},
        files={
            "request": (None, json.dumps({"model": MODEL, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}),
                        "application/json"),
            "audio": ("question.wav", wav, "audio/wav"),
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("transcript", "").strip()
