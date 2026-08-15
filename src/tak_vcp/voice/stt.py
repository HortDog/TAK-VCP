"""Whisper transcription (faster-whisper) and closed-vocabulary matching.

Production role (step 5): transcribe only the free-text tail of parameterized
commands, after the classifier has resolved the verb. Interim role (until
command models are trained): transcribe the armed window and snap it to the
closed vocabulary with match_command — never dispatching outside the table.
"""

import difflib
import re

import numpy as np

from ..commands import COMMANDS


class TailTranscriber:
    """CPU int8 faster-whisper transcriber for short command audio."""

    def __init__(self, model_size: str = "base.en"):
        # Deferred import: heavy, and downloads the model on first use.
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(audio, language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def match_command(text: str, min_ratio: float = 0.75) -> "tuple[str, float] | None":
    """Snap free text to the closed command vocabulary.

    Compares against each phrase both whole and prefix-length (so a trailing
    argument like "... bearing zero four five" doesn't sink the match). Returns
    (phrase, ratio) or None — by construction never anything outside COMMANDS.
    """
    normalized = _normalize(text)
    if not normalized:
        return None
    best_phrase, best_ratio = None, 0.0
    for phrase in COMMANDS:
        whole = difflib.SequenceMatcher(None, normalized, phrase).ratio()
        prefix = difflib.SequenceMatcher(None, normalized[: len(phrase)], phrase).ratio()
        ratio = max(whole, prefix)
        if ratio > best_ratio:
            best_phrase, best_ratio = phrase, ratio
    if best_phrase is None or best_ratio < min_ratio:
        return None
    return best_phrase, best_ratio
