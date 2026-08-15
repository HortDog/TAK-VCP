"""Stub satisfying openwakeword train.py's unconditional Piper import.

train.py does `from generate_samples import generate_samples` even when only
--augment_clips/--train_model are requested. Piper's sample generator isn't
Windows-friendly, so configs point piper_sample_generator_path here; actual
clip generation is training/generate_samples.py (edge-tts).
"""


def generate_samples(*args, **kwargs):
    raise NotImplementedError(
        "Piper generation is stubbed out on this setup - generate clips with "
        "`uv run python training/generate_samples.py` instead of --generate_clips"
    )
