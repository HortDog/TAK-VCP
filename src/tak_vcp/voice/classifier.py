"""Layer 2: closed-set command classifier (not yet built).

One trained openwakeword model per phrase in tak_vcp.commands.COMMANDS; fires
on streaming ~80 ms frames, so bare commands dispatch with no STT pass in the
path. Emits per-command confidence scores for thresholding and
confirmation-trigger policy. Cannot emit anything outside the command table —
that closed-set property is the point; don't route the verb through Whisper.

Open items: per-command thresholds, cross-activation testing across the
vocabulary, training data (synthetic TTS bootstrap, then real operator
recordings under stress).
"""


class CommandClassifier:
    """Resolves an armed utterance to a phrase from the command vocabulary."""

    def __init__(self, model_dir: str, thresholds: dict[str, float] | None = None):
        raise NotImplementedError(
            "Build order step 4: train per-command openwakeword models and "
            "wire the classifier (see docs/voice-to-cot-handoff.md)"
        )
