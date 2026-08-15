"""Single source of the model list to train: wake phrase + command vocabulary.

Adversarial negatives per model teach the classifier what NOT to fire on:
near-collisions for the wake phrase, the *other* commands for each command
(directly attacks the cross-activation problem flagged in the handoff doc).
"""

from tak_vcp.commands import COMMANDS
from tak_vcp.voice.classifier import model_stem
from tak_vcp.voice.wake import TARGET_WAKE_PHRASE

WAKE_ADVERSARIAL = [
    "activate",
    "tak",
    "attack",
    "active tak",
    "activate track",
    "activate that",
]


def model_specs() -> "list[dict]":
    specs = [
        {
            "name": model_stem(TARGET_WAKE_PHRASE),
            "phrase": TARGET_WAKE_PHRASE,
            "negatives": WAKE_ADVERSARIAL,
        }
    ]
    for phrase in COMMANDS:
        specs.append(
            {
                "name": model_stem(phrase),
                "phrase": phrase,
                "negatives": [p for p in COMMANDS if p != phrase],
            }
        )
    return specs
