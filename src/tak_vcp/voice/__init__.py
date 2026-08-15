"""Voice front end: two-layer wake word + command classifier design (not yet built).

Layer 1 (wake.py) arms the system on the activation phrase. Layer 2
(classifier.py) resolves the utterance against the closed command vocabulary in
tak_vcp.commands. Whisper (stt.py) transcribes only the free-text tail of
parameterized commands — never the command verb itself. See
docs/voice-to-cot-handoff.md, "Design decision" section, for why.
"""
