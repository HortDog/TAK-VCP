"""Send a hardcoded CoT marker — step 3 of the build order, no voice involved.

Isolates transport/formatting issues before any audio complexity is added:

    uv run tak-vcp-send-test
    uv run tak-vcp-send-test --command "mark friendly" --lat 38.8895 --lon -77.0353
    uv run tak-vcp-send-test --cot-url tcp://192.168.1.50:8087
"""

import argparse
import asyncio
from configparser import ConfigParser

import pytak

from .commands import COMMANDS
from .cot import build_marker_cot
from .transport import DEFAULT_COT_URL


async def send_one(cot_url: str, event: bytes) -> None:
    """Open the PyTAK protocol for cot_url, send a single event, close."""
    config = ConfigParser()
    config["tak-vcp"] = {"COT_URL": cot_url}
    _reader, writer = await pytak.protocol_factory(config["tak-vcp"])

    # Mirror pytak's TXWorker handling: datagram writers expose an async
    # send(); stream writers use write()/drain().
    if hasattr(writer, "send"):
        await writer.send(event)
    else:
        writer.write(event)
        if hasattr(writer, "drain"):
            await writer.drain()

    if hasattr(writer, "close"):
        writer.close()
    if hasattr(writer, "wait_closed"):
        await writer.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one hardcoded CoT marker to a TAK client (no voice)."
    )
    parser.add_argument(
        "--cot-url",
        default=DEFAULT_COT_URL,
        help="PyTAK COT_URL: udp://<multicast>:<port> or tcp://<host>:<port> "
        f"(default: {DEFAULT_COT_URL})",
    )
    parser.add_argument(
        "--command",
        default="drop waypoint",
        choices=sorted(COMMANDS),
        help="voice command to simulate (default: drop waypoint)",
    )
    parser.add_argument("--lat", type=float, default=0.0, help="marker latitude")
    parser.add_argument("--lon", type=float, default=0.0, help="marker longitude")
    parser.add_argument(
        "--callsign", default=None, help="marker callsign (default: the UID)"
    )
    args = parser.parse_args()

    event = build_marker_cot(args.command, args.lat, args.lon, callsign=args.callsign)
    print(event.decode())
    asyncio.run(send_one(args.cot_url, event))
    print(
        f"\nSent '{args.command}' marker to {args.cot_url} "
        f"at lat={args.lat} lon={args.lon}."
    )
    if not args.lat and not args.lon:
        print("Note: marker is at 0,0 (Null Island) — pass --lat/--lon for your AO.")


if __name__ == "__main__":
    main()
