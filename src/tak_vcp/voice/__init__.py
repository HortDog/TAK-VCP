"""Voice front end: two-layer wake word + command classifier design.

Layer 1 (wake.py) arms the system on the activation phrase ("TAK active";
custom model pending training, "hey_jarvis" stand-in meanwhile). Layer 2
(classifier.py) resolves the utterance against the closed vocabulary in
tak_vcp.commands. Whisper (stt.py) transcribes free-text argument tails —
and, as an interim mode only, whole commands snapped to the closed set.
pipeline.py orchestrates and dispatches CoT events; audio.py owns the mic.

See docs/voice-to-cot-handoff.md ("Design decision" section) for why the verb
is classified, not transcribed.
"""
