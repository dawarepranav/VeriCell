import httpx
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
key = next(l.split("=", 1)[1].strip().strip('"') for l in (ROOT / ".env").read_text().splitlines() if l.startswith("ELEVENLABS_API_KEY="))
prompt = ("Energetic modern tech product launch music loop, driving electronic drums and pulsing bass, bright synth arpeggio, "
          "124 BPM, upbeat and confident, instrumental")
for i in range(2):
    r = httpx.post("https://api.elevenlabs.io/v1/sound-generation", headers={"xi-api-key": key}, timeout=120,
                   params={"output_format": "mp3_44100_128"},
                   json={"text": prompt, "duration_seconds": 22, "prompt_influence": 0.6, "loop": True, "model_id": "eleven_text_to_sound_v2"})
    print(i, r.status_code, r.text[:200] if r.status_code != 200 else len(r.content))
    if r.status_code == 200:
        Path(f"public/audio/loop{i}.mp3").write_bytes(r.content)
