"""CoT event builder: maps a recognized command to a CoT XML event."""

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .commands import COMMANDS, DEFAULT_COT_TYPE

COT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# How long the marker persists before TAK considers it expired — tune per use case.
DEFAULT_STALE_MINUTES = 5.0


def cot_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(COT_TIME_FORMAT)


def build_marker_cot(
    command: str,
    lat: float,
    lon: float,
    *,
    callsign: str | None = None,
    uid: str | None = None,
    stale_minutes: float = DEFAULT_STALE_MINUTES,
) -> bytes:
    """Build a CoT marker event for a recognized voice command.

    Unknown commands fall back to a generic waypoint rather than guessing an
    affiliation. UIDs are random rather than timestamp-derived so a wake-word
    double-fire can't collide.
    """
    cmd = COMMANDS.get(command)
    cot_type = cmd.cot_type if cmd else DEFAULT_COT_TYPE

    now = datetime.now(timezone.utc)
    uid = uid or f"voice-{uuid.uuid4().hex[:12]}"

    event = ET.Element(
        "event",
        {
            "version": "2.0",
            "uid": uid,
            "type": cot_type,
            "how": "m-g",  # manually generated
            "time": cot_time(now),
            "start": cot_time(now),
            "stale": cot_time(now + timedelta(minutes=stale_minutes)),
        },
    )
    ET.SubElement(
        event,
        "point",
        {"lat": str(lat), "lon": str(lon), "hae": "0", "ce": "10", "le": "10"},
    )
    detail = ET.SubElement(event, "detail")
    ET.SubElement(detail, "contact", {"callsign": callsign or uid})

    return ET.tostring(event)
