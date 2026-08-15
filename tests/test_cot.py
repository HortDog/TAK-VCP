import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest

from tak_vcp.commands import COMMANDS
from tak_vcp.cot import COT_TIME_FORMAT, build_marker_cot


@pytest.mark.parametrize("command,expected_type", [(c.phrase, c.cot_type) for c in COMMANDS.values()])
def test_command_maps_to_cot_type(command, expected_type):
    event = ET.fromstring(build_marker_cot(command, 38.0, -77.0))
    assert event.get("type") == expected_type


def test_unknown_command_falls_back_to_waypoint():
    event = ET.fromstring(build_marker_cot("do a barrel roll", 0.0, 0.0))
    assert event.get("type") == "b-m-p-w"


def test_event_structure():
    event = ET.fromstring(build_marker_cot("drop waypoint", 38.8895, -77.0353))
    assert event.get("version") == "2.0"
    assert event.get("how") == "m-g"

    point = event.find("point")
    assert point.get("lat") == "38.8895"
    assert point.get("lon") == "-77.0353"

    start = datetime.strptime(event.get("start"), COT_TIME_FORMAT).replace(tzinfo=timezone.utc)
    stale = datetime.strptime(event.get("stale"), COT_TIME_FORMAT).replace(tzinfo=timezone.utc)
    assert (stale - start).total_seconds() == pytest.approx(300, abs=1)


def test_uids_do_not_collide_on_double_fire():
    a = ET.fromstring(build_marker_cot("drop waypoint", 0.0, 0.0))
    b = ET.fromstring(build_marker_cot("drop waypoint", 0.0, 0.0))
    assert a.get("uid") != b.get("uid")


def test_callsign_with_xml_special_chars_is_escaped():
    xml = build_marker_cot("drop waypoint", 0.0, 0.0, callsign='B&B <"Alpha">')
    event = ET.fromstring(xml)
    assert event.find("detail/contact").get("callsign") == 'B&B <"Alpha">'
