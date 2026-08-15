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

DEFAULT_COT_URL = "udp://239.2.3.1:6969"


class VoiceCommandWorker(pytak.QueueWorker):
    """Drains the voice pipeline's output queue into PyTAK's transmit queue."""

    def __init__(self, queue, config, voice_queue: "asyncio.Queue[bytes]"):
        super().__init__(queue, config)
        self.voice_queue = voice_queue

    async def handle_data(self, data: bytes) -> None:
        await self.put_queue(data)

    async def run(self, number_of_iterations: int = -1) -> None:
        while True:
            cot_xml = await self.voice_queue.get()
            await self.handle_data(cot_xml)


async def run_sender(
    voice_queue: "asyncio.Queue[bytes]", cot_url: str = DEFAULT_COT_URL
) -> None:
    """Run the PyTAK client forever, transmitting anything put on voice_queue."""
    config = ConfigParser()
    config["tak-vcp"] = {"COT_URL": cot_url}
    section = config["tak-vcp"]

    clitool = pytak.CLITool(section)
    await clitool.setup()
    clitool.add_tasks({VoiceCommandWorker(clitool.tx_queue, section, voice_queue)})
    await clitool.run()
