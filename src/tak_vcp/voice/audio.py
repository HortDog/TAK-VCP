"""Mic capture: blocking iterator of 80 ms int16 mono frames at 16 kHz.

80 ms / 1280 samples is openwakeword's native frame size; both voice layers
consume these frames directly.
"""

import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms


class MicFrames:
    """Context manager + iterator yielding int16 frames from an input device."""

    def __init__(self, device: int | None = None):
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=device,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        self._q.put(indata[:, 0].copy())

    def __enter__(self) -> "MicFrames":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def __iter__(self) -> "MicFrames":
        return self

    def __next__(self) -> np.ndarray:
        return self._q.get()


def list_input_devices() -> str:
    return str(sd.query_devices())
