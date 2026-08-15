"""Send a hardcoded CoT marker — step 3 of the build order, no voice involved.

Isolates transport/formatting issues before any audio complexity is added:

    uv run tak-vcp-send-test
    uv run tak-vcp-send-test --command "mark friendly" --lat -23.698 --lon 133.881
    uv run tak-vcp-send-test --cot-url tcp://127.0.0.1:8087   # WinTAK on this host
    uv run tak-vcp-send-test --cot-url tcp://192.168.1.50:8087

Same-host note: use the TCP form — WinTAK listens on TCP 4242/8087 by default,
and loopback multicast does not deliver to it (see README transport notes).
"""

import argparse
import asyncio

from .commands import COMMANDS
from .cot import build_marker_cot
from .transport import DEFAULT_COT_URL, send_one


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
    parser.add_argument(
        "--local-addr",
        default=None,
        help="local interface IP to send multicast from (needed on multi-homed "
        "hosts, e.g. when a VPN adapter would otherwise win the route)",
    )
    args = parser.parse_args()

    event = build_marker_cot(args.command, args.lat, args.lon, callsign=args.callsign)
    print(event.decode())
    asyncio.run(send_one(args.cot_url, event, local_addr=args.local_addr))
    print(
        f"\nSent '{args.command}' marker to {args.cot_url} "
        f"at lat={args.lat} lon={args.lon}."
    )
    if not args.lat and not args.lon:
        print("Note: marker is at 0,0 (Null Island) — pass --lat/--lon for your AO.")


if __name__ == "__main__":
    main()
