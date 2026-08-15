"""Layer 1: wake word listener — arms the command layer on the activation phrase.

Target phrase is "TAK active" (custom openwakeword model, training pending —
see docs/training.md). Until models/tak_active.onnx exists, the pretrained
"hey_jarvis" model serves as a stand-in so the pipeline can be exercised.
"""

import openwakeword
import openwakeword.utils
from openwakeword.model import Model

TARGET_WAKE_PHRASE = "tak active"
STAND_IN_MODEL = "hey_jarvis"


def ensure_shared_models(pretrained_name: str = STAND_IN_MODEL) -> None:
    """Download openwakeword's shared melspectrogram/embedding models (and the
    named pretrained wake model) into the package cache on first run."""
    openwakeword.utils.download_models(model_names=[pretrained_name])


class WakeWordListener:
    """Scores each 80 ms frame against one wake model; caller applies arming logic."""

    def __init__(self, model: str = STAND_IN_MODEL, threshold: float = 0.5):
        self.threshold = threshold
        if model in openwakeword.MODELS:
            ensure_shared_models(model)
        else:
            # Custom .onnx path — shared feature models must still exist.
            ensure_shared_models()
        self.model = Model(wakeword_models=[model], inference_framework="onnx")
        self.key = next(iter(self.model.models))

    def score(self, frame) -> float:
        return float(self.model.predict(frame)[self.key])

    def triggered(self, frame) -> bool:
        return self.score(frame) >= self.threshold

    def reset(self) -> None:
        """Clear streaming buffers so residual audio can't re-trigger."""
        self.model.reset()
