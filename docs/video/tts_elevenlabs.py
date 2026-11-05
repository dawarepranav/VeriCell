"""Voice-over with ElevenLabs: one clip per scene from ELEVENLABS.md, saved where render_video.py reads them.

    python docs/video/tts_elevenlabs.py voices          # list the voices on your account
    python docs/video/tts_elevenlabs.py [VOICE_ID]      # generate the 12 clips, then re-run render_video.py

Reads ELEVENLABS_API_KEY from the repo's .env (never printed).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BUILD = HERE / "build"
API = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE = "nPczCjzI2devNBz1zQrb"  # "Brian": calm, warm narrator from the ElevenLabs library


def key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ELEVENLABS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("Add ELEVENLABS_API_KEY=... to .env first")


def scenes() -> dict[str, str]:
    """Scene id -> text, from the '## id' sections of ELEVENLABS.md."""
    text = (HERE / "ELEVENLABS.md").read_text()
    out = {}
    for m in re.finditer(r"^## (\w+)\n(.+?)(?=\n## |\Z)", text, flags=re.S | re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout)


def main() -> None:
    h = {"xi-api-key": key()}
    if len(sys.argv) > 1 and sys.argv[1] == "voices":
        for v in httpx.get(f"{API}/voices", headers=h, timeout=60).json()["voices"]:
            print(v["voice_id"], v["name"], v.get("labels", {}).get("description", ""))
        return
    voice = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOICE
    timed = json.loads((BUILD / "scenes_timed.json").read_text())
    texts = scenes()
    (BUILD / "vo").mkdir(parents=True, exist_ok=True)
    for s in timed:
        mp3 = BUILD / "vo" / f"{s['id']}.mp3"
        r = httpx.post(f"{API}/text-to-speech/{voice}", headers=h, timeout=120,
                       params={"output_format": "mp3_44100_128"},
                       json={"text": texts[s["id"]], "model_id": "eleven_multilingual_v2",
                             "voice_settings": {"stability": 0.38, "similarity_boost": 0.75, "style": 0.4,
                                                "use_speaker_boost": True, "speed": 1.12}})
        if r.status_code != 200:
            sys.exit(f"{s['id']}: ElevenLabs returned {r.status_code}: {r.text[:300]}")
        mp3.write_bytes(r.content)
        # render_video.py reads build/vo/<id>.aiff
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, BUILD / "vo" / f"{s['id']}.aiff"], check=True)
        s["vo_seconds"] = round(duration(mp3), 2)
        s["vo"] = re.sub(r"<break[^>]*/>\s*", "", texts[s["id"]])  # captions without the pause tags
        print(f"{s['id']:10} {s['vo_seconds']:5.1f}s", flush=True)
    (BUILD / "scenes_timed.json").write_text(json.dumps(timed, indent=1))
    print("total voice", round(sum(s["vo_seconds"] for s in timed), 1), "s; now run render_video.py")


if __name__ == "__main__":
    main()
