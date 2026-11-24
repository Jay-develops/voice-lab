"""Voice lab: speak one script with several free voices, line by line, for a hand-made video.

Usage: python voicelab/make.py voicelab/<script>.json kokoro|chatterbox|compare
Writes voicelab/out/<voice>/<line id>.wav, voicelab/out/<engine>.json (duration and word timings
per line) and voicelab/out/compare.wav (line 1 in every voice, one after another).
Nothing here is posted anywhere.
"""
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
PART = os.environ.get("PART", "*").strip() or "*"     # voice only the lines whose id starts with this
SUFFIX = "" if PART == "*" else f"-{PART}"


def lines_of(script):
    return [l for l in script["lines"] if PART == "*" or l["id"].startswith(PART)]


def write_wav(path, x, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767
    return x, sr


def kokoro(script):
    if not script["kokoro"]:
        return
    from kokoro import KPipeline
    pipes, info = {}, {}
    for voice, speed in script["kokoro"]:
        lang = voice[0]
        if lang not in pipes:
            pipes[lang] = KPipeline(lang_code=lang)
        info[voice] = {}
        for line in lines_of(script):
            parts, words, offset = [], [], 0.0
            for res in pipes[lang](line["say"], voice=voice, speed=speed):
                audio = getattr(res, "audio", None)
                if audio is None:
                    continue
                a = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
                for tok in (getattr(res, "tokens", None) or []):
                    if tok.start_ts is not None and tok.end_ts is not None and tok.text.strip():
                        words.append([tok.text, round(offset + tok.start_ts, 3), round(offset + tok.end_ts, 3)])
                parts.append(np.asarray(a, dtype=np.float32))
                offset += len(a) / 24000
            x = np.concatenate(parts)
            write_wav(OUT / voice / f"{line['id']}.wav", x, 24000)
            info[voice][line["id"]] = {"dur": round(len(x) / 24000, 3), "words": words}
            print(f"{voice} {line['id']} {len(x) / 24000:5.2f}s  {line['say']}", flush=True)
    (OUT / f"kokoro{SUFFIX}.json").write_text(json.dumps(info, indent=1))


def chatterbox(script):
    if not script["chatterbox"]:
        return
    import torch
    real_load = torch.load
    torch.load = lambda *a, **k: real_load(*a, **{**k, "map_location": k.get("map_location") or "cpu"})
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cpu")
    info = {}
    for name, exaggeration, cfg in script["chatterbox"]:
        info[name] = {}
        for line in lines_of(script):
            torch.manual_seed(7)
            wav = model.generate(line["say"], exaggeration=line.get("exaggeration", exaggeration),
                                 cfg_weight=line.get("cfg", cfg))
            x = wav.squeeze(0).detach().cpu().numpy()
            write_wav(OUT / name / f"{line['id']}.wav", x, model.sr)
            info[name][line["id"]] = {"dur": round(len(x) / model.sr, 3), "sr": model.sr, "words": []}
            print(f"{name} {line['id']} {len(x) / model.sr:5.2f}s  {line['say']}", flush=True)
    (OUT / f"chatterbox{SUFFIX}.json").write_text(json.dumps(info, indent=1))


def compare(script):
    first = lines_of(script)[0]["id"]
    out, sr0 = [], 24000
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        f = d / f"{first}.wav"
        if not f.exists():
            continue
        x, sr = read_wav(f)
        if sr != sr0:
            x = np.interp(np.arange(0, len(x), sr / sr0), np.arange(len(x)), x).astype(np.float32)
        out += [x, np.zeros(int(sr0 * 0.8), np.float32)]
        print(f"compare: {d.name}", flush=True)
    if out:
        write_wav(OUT / "compare.wav", np.concatenate(out), sr0)


if __name__ == "__main__":
    script = json.loads(Path(sys.argv[1]).read_text())
    {"kokoro": kokoro, "chatterbox": chatterbox, "compare": compare}[sys.argv[2]](script)
