"""Fetch the training datasets openwakeword's augment/train stages need.

- Precomputed negative features (ACAV100M, ~5.8 GB) + validation features —
  the general speech/noise negatives that keep false-positive rates down.
- MIT environmental room impulse responses (~50 MB) for reverb augmentation.
- Background noise: BYO. Drop 16 kHz wavs of the real operating environment
  (vehicle, wind, radio chatter) into training/data/background/ — that's
  better training signal than any stock dataset. --placeholder-noise writes
  synthetic white/pink beds so the pipeline runs before you have recordings.

    uv run python training/download_data.py --placeholder-noise
"""

import argparse
import wave
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
FEATURES_REPO = "davidscripka/openwakeword_features"
FEATURE_FILES = [
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "validation_set_features.npy",
]
RIR_REPO = "davidscripka/MIT_environmental_impulse_responses"


def download_features() -> None:
    from huggingface_hub import hf_hub_download

    for name in FEATURE_FILES:
        if (DATA / name).exists():
            print(f"{name}: already present")
            continue
        print(f"downloading {name} (the ACAV100M file is ~5.8 GB — go make coffee)")
        hf_hub_download(
            repo_id=FEATURES_REPO, filename=name, repo_type="dataset",
            local_dir=DATA,
        )


def download_rirs() -> None:
    out = DATA / "rirs"
    if out.exists() and any(out.glob("*.wav")):
        print("rirs: already present")
        return
    from huggingface_hub import snapshot_download

    # The repo stores raw 16 kHz wavs under 16khz/; flatten them into rirs/
    # because train.py scans rir_paths non-recursively.
    staging = DATA / "_rir_download"
    snapshot_download(
        repo_id=RIR_REPO, repo_type="dataset", local_dir=staging,
        allow_patterns=["16khz/*.wav"],
    )
    out.mkdir(parents=True, exist_ok=True)
    for wav in (staging / "16khz").glob("*.wav"):
        wav.replace(out / wav.name)
    print(f"rirs: {len(list(out.glob('*.wav')))} files")


def placeholder_noise() -> None:
    out = DATA / "background"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        white = rng.normal(0, 1, 16000 * 60)
        # crude pink-ish tilt via cumulative smoothing on odd indices
        if i % 2:
            white = np.convolve(white, np.ones(8) / 8, mode="same")
        samples = (white / np.abs(white).max() * 8000).astype(np.int16)
        with wave.open(str(out / f"placeholder_noise_{i}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(samples.tobytes())
    print(f"background: wrote placeholder beds into {out} — replace with real "
          "operating-environment recordings before trusting a model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--placeholder-noise", action="store_true")
    parser.add_argument("--skip-features", action="store_true", help="skip the 5.8 GB download")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if not args.skip_features:
        download_features()
    download_rirs()
    if args.placeholder_noise:
        placeholder_noise()


if __name__ == "__main__":
    main()
