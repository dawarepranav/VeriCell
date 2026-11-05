"""The film's five-act score as one audio bed, placed on the real timeline (public/audio/score.wav)."""
import json, subprocess, wave
import numpy as np

R = 44100
tl = json.load(open("src/timeline.json")); S = {s["id"]: s for s in tl["scenes"]}; T = tl["total"]

def stem(name, start=0.0, lowpass=None, target=-15.0):
    af = ["-af", f"lowpass=f={lowpass}"] if lowpass else []
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", f"public/audio/{name}.mp3", *af, "-ac", "1", "-ar", str(R),
                          "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768
    x = x * 10 ** ((target - 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)) / 20)
    return x

t = np.arange(int((T + 0.5) * R)) / R
def env(points):  # piecewise-linear envelope [(time, level), ...]
    return np.interp(t, [p[0] for p in points], [p[1] for p in points])
def fit(x):
    return np.resize(x, len(t))

hook_end = S["hook"]["vo_at"] + S["hook"]["vo_len"]
act2 = S["pitch"]["start"] + 0.1
act3 = S["verdict"]["start"] + 3.0
checks = S["checks"]["start"]
ding = checks + next(x["t"] for x in S["checks"]["sfx"] if "ding" in x["src"])
reveal = S["truth"]["start"] + 2.1     # the "<1%" card
close_end = S["close"]["start"] + S["close"]["vo_at"] + S["close"]["vo_len"]

drone = fit(stem("dawn", 0, target=-17))            # cold, beatless
pulse = fit(stem("ouroboros", 0, lowpass=1200))     # tense, filtered pulse
pad = fit(stem("dawn", 215, target=-17))            # suspended
warm = fit(stem("warm", 0, lowpass=3500, target=-16))

score = (drone * env([(0, 0), (1.0, .9), (hook_end - .4, .9), (hook_end + .1, 0), (T + 1, 0)])          # act 1, then the vacuum
         + pulse * env([(0, 0), (act2 - .01, 0), (act2 + .3, 1), (act3, 1), (act3 + 1.0, 0), (T + 1, 0)])  # act 2
         + pad * env([(0, 0), (act3, 0), (act3 + 1.0, 1), (checks, 1), (checks + .6, .4),                  # act 3, then near-silent clicks
                      (ding - .55, .4), (ding - .5, 0), (T + 1, 0)])                                       # dead stop before the chime
         + warm * env([(0, 0), (reveal - .05, 0), (reveal + .9, 1), (close_end, 1), (close_end + .5, 1.7), (T - .1, 0), (T + 1, 0)]))  # bloom, ride, swell
score = np.clip(score, -1, 1)
with wave.open("public/audio/score.wav", "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(R); w.writeframes((score * 32767).astype(np.int16).tobytes())
print(f"acts: drone to {hook_end:.1f}s · pulse from {act2:.1f}s · suspension {act3:.1f}s · clicks {checks:.1f}s · stop {ding-.5:.1f}s · chime {ding:.1f}s · bloom {reveal:.1f}s · swell {close_end:.1f}s · end {T:.1f}s")
