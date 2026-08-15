"""Listen on the CoT multicast group and print arriving packets.

Build-order step 2 helper: if WinTAK isn't showing markers, this tells you
whether the problem is transport (nothing on the wire) or WinTAK-side config
(packets arriving but not displayed). Note: WinTAK binds 6969 exclusively
(SO_EXCLUSIVEADDRUSE), so on the same host run this while WinTAK is stopped,
or listen on a different --port.

    uv run tak-vcp-listen
    uv run tak-vcp-listen --max 1 --timeout 20

Exits 0 if at least one packet arrived, 1 otherwise.
"""

import argparse
import socket
import struct
import sys

from .transport import DEFAULT_COT_URL

DEFAULT_GROUP = "239.2.3.1"
DEFAULT_PORT = 6969


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen on a UDP multicast group and print CoT packets."
    )
    parser.add_argument("--group", default=DEFAULT_GROUP, help="multicast group")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds of silence before giving up (default: 30)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        dest="max_packets",
        help="exit after N packets (default: 0 = listen until timeout)",
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.port))
    membership = struct.pack(
        "4s4s", socket.inet_aton(args.group), socket.inet_aton("0.0.0.0")
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sock.settimeout(args.timeout)

    print(f"Listening on udp://{args.group}:{args.port} (timeout {args.timeout:g}s)")
    received = 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            received += 1
            print(f"\n[{received}] {len(data)} bytes from {addr[0]}:{addr[1]}")
            print(data.decode("utf-8", errors="replace"))
            if args.max_packets and received >= args.max_packets:
                break
    except KeyboardInterrupt:
        pass

    print(f"\n{received} packet(s) received on udp://{args.group}:{args.port}")
    if not received:
        print(f"Nothing on the wire — is the sender pointed at {DEFAULT_COT_URL}?")
    sys.exit(0 if received else 1)


if __name__ == "__main__":
    main()
