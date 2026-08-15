"""Layer 1: wake word listener (not yet built).

openwakeword model listening continuously on the mic (sounddevice); on
activation, arms the layer-2 command classifier. Needs debounce so a
double-fire can't dispatch twice.
"""


class WakeWordListener:
    """Continuously listens for the activation phrase and arms the classifier."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        raise NotImplementedError(
            "Build order step 4: wire openwakeword layer-1 listener "
            "(see docs/voice-to-cot-handoff.md)"
        )
