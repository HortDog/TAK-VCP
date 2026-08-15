"""PyTAK transport: forwards CoT events from the voice pipeline to the TAK network.

PyTAK handles serialization and delivery so the voice pipeline never touches
raw sockets. Transport options via COT_URL:

- ``udp://239.2.3.1:6969`` — multicast; default WinTAK listens here when both
  machines share a LAN/VLAN, no WinTAK-side config needed.
- ``tcp://<wintak-host>:8087`` — point-to-point; requires a matching TCP input
  in WinTAK's network preferences.
- TAK Server URL — WinTAK subscribes through its normal server connection.
"""

import asyncio
from configparser import ConfigParser

import pytak

# +wo (write-only) matters: without it pytak also binds a receive socket on
# the port, which fails with WinError 10013 when WinTAK on the same host
# already holds 6969. This pipeline only ever transmits.
DEFAULT_COT_URL = "udp+wo://239.2.3.1:6969"


async def send_one(cot_url: str, event: bytes, local_addr: str | None = None) -> None:
    """Open the PyTAK protocol for cot_url, send a single event, close."""
    config = ConfigParser()
    config["tak-vcp"] = {"COT_URL": cot_url}
    if local_addr:
        # Pin the multicast egress NIC on multi-homed hosts (VPN adapters).
        config["tak-vcp"]["PYTAK_MULTICAST_LOCAL_ADDR"] = local_addr
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


async def run_sender(
    voice_queue: "asyncio.Queue[bytes]",
    cot_url: str = DEFAULT_COT_URL,
    local_addr: str | None = None,
) -> None:
    """Transmit each event from voice_queue on its own short-lived connection.

    One connection per event is deliberate: WinTAK's default TCP inputs parse
    a connection's data only when the connection CLOSES (one-shot semantics),
    so events written to a persistent stream never render. Verified empirically
    — a held-open pytak connection delivered bytes that WinTAK sat on. UDP is
    unaffected either way (datagrams are self-delimiting).
    """
    while True:
        event = await voice_queue.get()
        try:
            await send_one(cot_url, event, local_addr=local_addr)
        except OSError as exc:
            print(f"[transport] send failed ({exc}); event dropped", flush=True)
