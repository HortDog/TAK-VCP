"""Whisper tail transcription for parameterized commands only (not yet built).

Invoked after the classifier has already resolved the verb, to transcribe the
free-text tail: bearing/range, labels, callsigns. E.g. "mark contact bearing
zero four five range two hundred" — the classifier resolves "mark contact"
(safety-relevant); this module handles the tail, where a misrecognition is
cosmetic rather than tactical. Requires a confirmation step before dispatch
for anything that sets affiliation/type.
"""


class TailTranscriber:
    """Transcribes the argument tail of a parameterized command via faster-whisper."""

    def __init__(self, model_size: str = "base.en"):
        raise NotImplementedError(
            "Build order step 5: add faster-whisper tail transcription and the "
            "argument parser (see docs/voice-to-cot-handoff.md)"
        )
