"""Generate positive + adversarial-negative training clips with Edge neural TTS.

Windows-friendly replacement for openwakeword's Piper stage: writes 16 kHz
mono wavs into the exact layout train.py's --augment_clips stage expects
(training/output/<model>/{positive,negative}_{train,test}/). Varies voice,
rate, and pitch per clip. Needs network (Edge TTS is a web service).

    uv run python training/generate_samples.py --all --n 400
    uv run python training/generate_samples.py --model activate_tak --n 50
"""

import argparse
import asyncio
import io
import itertools
import random
import wave
from pathlib import Path

import av
import edge_tts
import numpy as np

from phrases import model_specs

OUTPUT = Path(__file__).resolve().parent / "output"

VOICES = [
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural", "en-US-ChristopherNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural",
    "en-CA-LiamNeural", "en-CA-ClaraNeural", "en-IN-PrabhatNeural",
]
RATES = ["-25%", "-10%", "+0%", "+15%", "+30%"]
PITCHES = ["-30Hz", "-15Hz", "+0Hz", "+15Hz", "+30Hz"]


def variations(seed: int):
    combos = list(itertools.product(VOICES, RATES, PITCHES))
    random.Random(seed).shuffle(combos)
    return itertools.cycle(combos)


async def synth(text: str, voice: str, rate: str, pitch: str) -> bytes:
    stream = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    audio = b""
    async for chunk in stream.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio


def to_16k_mono(mp3_bytes: bytes) -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(io.BytesIO(mp3_bytes)) as container:
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray() for f in resampler.resample(frame)]
    chunks += [f.to_ndarray() for f in resampler.resample(None)]
    return np.concatenate([c[0] for c in chunks if c.size])


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.astype(np.int16).tobytes())


async def generate(texts: "list[str]", out_dir: Path, n: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))
    if existing >= n:
        print(f"  {out_dir.name}: {existing} clips already present, skipping")
        return
    combo = variations(seed)
    text_cycle = itertools.cycle(texts)
    for i in range(existing, n):
        voice, rate, pitch = next(combo)
        text = next(text_cycle)
        try:
            audio = await synth(text, voice, rate, pitch)
            write_wav(out_dir / f"{i:05d}.wav", to_16k_mono(audio))
        except Exception as exc:  # noqa: BLE001 — keep generating past flaky requests
            print(f"  {out_dir.name}: clip {i} failed ({exc}), continuing")
    print(f"  {out_dir.name}: {len(list(out_dir.glob('*.wav')))} clips")


async def run(specs: "list[dict]", n: int) -> None:
    n_val = max(1, n // 5)
    for spec in specs:
        print(f"[{spec['name']}] positives: {spec['phrase']!r}, "
              f"{len(spec['negatives'])} negative phrases")
        base = OUTPUT / spec["name"]
        await generate([spec["phrase"]], base / "positive_train", n, seed=1)
        await generate([spec["phrase"]], base / "positive_test", n_val, seed=2)
        await generate(spec["negatives"], base / "negative_train", n, seed=3)
        await generate(spec["negatives"], base / "negative_test", n_val, seed=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="generate for every model")
    parser.add_argument("--model", help="single model name (e.g. activate_tak)")
    parser.add_argument("--n", type=int, default=400, help="clips per class (val = n/5)")
    args = parser.parse_args()

    specs = model_specs()
    if not args.all:
        specs = [s for s in specs if s["name"] == args.model]
        if not specs:
            raise SystemExit(f"unknown --model; choices: {[s['name'] for s in model_specs()]}")
    asyncio.run(run(specs, args.n))


if __name__ == "__main__":
    main()
