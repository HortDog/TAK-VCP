"""The closed-set command vocabulary and its CoT type mappings.

This table is the single source of truth for what the pipeline can say. The
layer-2 classifier is trained on exactly these phrases and cannot emit anything
outside them. Keep new phrases acoustically distant from existing ones —
similar phrases cross-activate small classifier heads.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    phrase: str
    cot_type: str
    description: str
    # True when the command puts an affiliation/type on the shared map — these
    # are candidates for confirmation or classifier/Whisper cross-check before
    # dispatch (confirmation policy is still an open item).
    affiliation_setting: bool = False
    # True when the command accepts a free-text tail (bearing/range, labels,
    # callsigns) that must go through the Whisper transcription path.
    takes_args: bool = False


COMMANDS: dict[str, Command] = {
    "mark contact": Command(
        phrase="mark contact",
        cot_type="a-h-G",
        description="Hostile, ground",
        affiliation_setting=True,
    ),
    "mark friendly": Command(
        phrase="mark friendly",
        cot_type="a-f-G",
        description="Friendly, ground",
        affiliation_setting=True,
    ),
    "mark casualty": Command(
        phrase="mark casualty",
        cot_type="a-f-G-U-C",
        description="Casualty, friendly",
        affiliation_setting=True,
    ),
    "drop waypoint": Command(
        phrase="drop waypoint",
        cot_type="b-m-p-w",
        description="Generic waypoint",
    ),
}

DEFAULT_COT_TYPE = "b-m-p-w"
