"""Layer 2: closed-set command classifier — one openwakeword model per command.

Fires on streaming 80 ms frames, so bare commands dispatch with no STT pass in
the path. Cannot emit anything outside tak_vcp.commands.COMMANDS — that
closed-set property is the design's key safety feature.

Model files are found by naming convention: models/<phrase with underscores>.onnx
(e.g. "mark contact" -> models/mark_contact.onnx). Training: docs/training.md.
"""

from pathlib import Path

from openwakeword.model import Model

from ..commands import COMMANDS
from .wake import ensure_shared_models


def model_stem(phrase: str) -> str:
    return phrase.replace(" ", "_")


def find_command_models(models_dir: str | Path) -> "dict[str, Path]":
    """Map command phrases to trained .onnx files present in models_dir."""
    found = {}
    for phrase in COMMANDS:
        path = Path(models_dir) / f"{model_stem(phrase)}.onnx"
        if path.exists():
            found[phrase] = path
    return found


class CommandClassifier:
    """Emits per-command confidence per frame; resolves to the closed set only."""

    def __init__(self, model_paths: "dict[str, Path]", threshold: float = 0.5):
        if not model_paths:
            raise ValueError("no command models given — see docs/training.md")
        self.threshold = threshold
        # openwakeword keys loaded models by filename stem.
        self.phrase_by_key = {model_stem(p): p for p in model_paths}
        ensure_shared_models()
        self.model = Model(
            wakeword_models=[str(p) for p in model_paths.values()],
            inference_framework="onnx",
        )

    def scores(self, frame) -> "dict[str, float]":
        raw = self.model.predict(frame)
        return {self.phrase_by_key.get(k, k): float(v) for k, v in raw.items()}

    def best(self, frame) -> "tuple[str, float] | None":
        """Highest-scoring command for this frame, or None if under threshold."""
        scores = self.scores(frame)
        phrase, score = max(scores.items(), key=lambda kv: kv[1])
        return (phrase, score) if score >= self.threshold else None

    def reset(self) -> None:
        self.model.reset()
