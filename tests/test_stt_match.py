import pytest

from tak_vcp.voice.stt import match_command


@pytest.mark.parametrize(
    "heard,expected",
    [
        ("mark contact", "mark contact"),
        ("Mark Contact.", "mark contact"),
        ("mark friendly", "mark friendly"),
        ("drop waypoint", "drop waypoint"),
        ("drop a waypoint", "drop waypoint"),
        ("mark casualty", "mark casualty"),
        ("mark contact bearing zero four five range two hundred", "mark contact"),
    ],
)
def test_matches_closed_set(heard, expected):
    match = match_command(heard)
    assert match is not None
    assert match[0] == expected


@pytest.mark.parametrize("heard", ["", "   ", "what a nice day", "do a barrel roll"])
def test_rejects_non_commands(heard):
    assert match_command(heard) is None


def test_never_emits_outside_vocabulary():
    from tak_vcp.commands import COMMANDS

    for text in ("mark", "contact", "way point", "friendly mark"):
        match = match_command(text)
        assert match is None or match[0] in COMMANDS
