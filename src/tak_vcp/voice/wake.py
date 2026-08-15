"""Layer 1: wake word listener — arms the command layer on the activation phrase.

Target phrase is "activate TAK" (custom openwakeword model — training/ has the
pipeline). Until models/activate_tak.onnx exists, the pretrained "hey_jarvis"
model serves as a stand-in so the pipeline can be exercised. Multiple wake
models can run side by side (any of them triggers) — useful if one phrase
proves flaky in the field.
"""

import openwakeword
import openwakeword.utils
from openwakeword.model import Model

TARGET_WAKE_PHRASE = "activate tak"
STAND_IN_MODEL = "hey_jarvis"


def ensure_shared_models(pretrained_names: "list[str] | None" = None) -> None:
    """Download openwakeword's shared melspectrogram/embedding models (and any
    named pretrained wake models) into the package cache on first run."""
    openwakeword.utils.download_models(model_names=pretrained_names or [STAND_IN_MODEL])


class WakeWordListener:
    """Scores each 80 ms frame against the wake model(s); any over threshold triggers."""

    def __init__(self, models: "str | list[str]" = STAND_IN_MODEL, threshold: float = 0.5):
        names = [models] if isinstance(models, str) else list(models)
        self.threshold = threshold
        pretrained = [n for n in names if n in openwakeword.MODELS]
        ensure_shared_models(pretrained or None)
        self.model = Model(wakeword_models=names, inference_framework="onnx")
        self.keys = list(self.model.models)
        self.label = " | ".join(self.keys)

    def score(self, frame) -> float:
        scores = self.model.predict(frame)
        return max(float(scores[k]) for k in self.keys)

    def triggered(self, frame) -> bool:
        return self.score(frame) >= self.threshold

    def reset(self) -> None:
        """Clear streaming buffers so residual audio can't re-trigger."""
        self.model.reset()
