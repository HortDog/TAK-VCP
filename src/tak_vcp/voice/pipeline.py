"""Voice pipeline orchestrator: mic -> wake (layer 1) -> command (layer 2) -> CoT -> PyTAK.

Command-layer modes:

- classifier (default, the production design): trained per-command openwakeword
  models from --models-dir. Closed set, streaming, no STT in the dispatch path.
- --stt-commands (interim, until command models are trained): records the armed
  window, transcribes with faster-whisper, snaps to the closed vocabulary via
  match_command. Higher latency and STT in the loop — bring-up/testing only.

Marker position is a fixed --lat/--lon for now; the position policy (own
position, bearing/range offsets) is an open item in the handoff doc.

    uv run tak-vcp-voice --stt-commands --dry-run
    uv run tak-vcp-voice --stt-commands --cot-url tcp://127.0.0.1:8087 --lat -23.7 --lon 133.88
"""

import argparse
import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from ..commands import COMMANDS
from ..cot import build_marker_cot
from ..transport import DEFAULT_COT_URL, run_sender
from .audio import MicFrames, list_input_devices
from .classifier import CommandClassifier, find_command_models, model_stem
from .wake import STAND_IN_MODEL, TARGET_WAKE_PHRASE, WakeWordListener


class VoicePipeline:
    """State machine over mic frames: LISTENING -> ARMED -> (dispatch) -> COOLDOWN."""

    def __init__(
        self,
        wake: WakeWordListener,
        on_command,
        classifier: "CommandClassifier | None" = None,
        transcriber=None,
        arm_window: float = 4.0,
        cooldown: float = 2.0,
        device: "int | None" = None,
    ):
        if (classifier is None) == (transcriber is None):
            raise ValueError("exactly one of classifier/transcriber is required")
        self.wake = wake
        self.classifier = classifier
        self.transcriber = transcriber
        self.on_command = on_command
        self.arm_window = arm_window
        self.cooldown = cooldown
        self.device = device

    def run(self, stop: threading.Event) -> None:
        state = "listening"
        deadline = 0.0
        captured: "list[np.ndarray]" = []
        print(f"[voice] listening (wake model: {self.wake.key})", flush=True)
        with MicFrames(self.device) as frames:
            for frame in frames:
                if stop.is_set():
                    return
                now = time.monotonic()

                if state == "listening":
                    if self.wake.triggered(frame):
                        print("[voice] armed — say a command", flush=True)
                        state = "armed"
                        deadline = now + self.arm_window
                        captured = []
                        if self.classifier:
                            self.classifier.reset()

                elif state == "armed":
                    dispatched = False
                    if self.classifier:
                        hit = self.classifier.best(frame)
                        if hit:
                            self.on_command(*hit)
                            dispatched = True
                    else:
                        captured.append(frame)

                    if dispatched:
                        state = "cooldown"
                        deadline = now + self.cooldown
                    elif now >= deadline:
                        if self.transcriber and captured:
                            self._transcribe_and_dispatch(np.concatenate(captured))
                        else:
                            print("[voice] disarmed (no command heard)", flush=True)
                        state = "cooldown"
                        deadline = now + self.cooldown

                elif state == "cooldown" and now >= deadline:
                    self.wake.reset()
                    state = "listening"
                    print("[voice] listening", flush=True)

    def _transcribe_and_dispatch(self, audio: np.ndarray) -> None:
        from .stt import match_command

        text = self.transcriber.transcribe(audio)
        match = match_command(text)
        if match:
            self.on_command(*match)
        else:
            print(f"[voice] rejected: {text!r} matches no command", flush=True)


async def _async_main(args) -> None:
    voice_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_command(phrase: str, confidence: float) -> None:
        event = build_marker_cot(phrase, args.lat, args.lon)
        print(f"[dispatch] {phrase} ({confidence:.2f})", flush=True)
        loop.call_soon_threadsafe(voice_queue.put_nowait, event)

    wake = WakeWordListener(args.wake_model, args.wake_threshold)
    classifier = transcriber = None
    if args.stt_commands:
        from .stt import TailTranscriber

        print(f"[voice] interim STT command mode ({args.stt_model})", flush=True)
        transcriber = TailTranscriber(args.stt_model)
    else:
        paths = find_command_models(args.models_dir)
        missing = [p for p in COMMANDS if p not in paths]
        if not paths:
            raise SystemExit(
                f"No command models in {args.models_dir}/ (expected e.g. "
                f"{model_stem(next(iter(COMMANDS)))}.onnx). Train them per "
                "docs/training.md, or use --stt-commands for the interim mode."
            )
        if missing:
            print(f"[voice] warning: no models for: {', '.join(missing)}", flush=True)
        classifier = CommandClassifier(paths, args.command_threshold)

    pipeline = VoicePipeline(
        wake,
        on_command,
        classifier=classifier,
        transcriber=transcriber,
        arm_window=args.arm_window,
        cooldown=args.cooldown,
        device=args.device,
    )
    stop = threading.Event()
    thread = threading.Thread(target=pipeline.run, args=(stop,), daemon=True)
    thread.start()
    try:
        if args.dry_run:
            print("[voice] dry run — events printed, not sent", flush=True)
            while True:
                event = await voice_queue.get()
                print(event.decode(), flush=True)
        else:
            await run_sender(voice_queue, args.cot_url, args.local_addr)
    finally:
        stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f'Voice-to-CoT pipeline (wake phrase target: "{TARGET_WAKE_PHRASE}").'
    )
    parser.add_argument(
        "--wake-model",
        default=None,
        help="pretrained openwakeword name or path to a trained .onnx "
        "(default: models/tak_active.onnx if present, else hey_jarvis stand-in)",
    )
    parser.add_argument("--wake-threshold", type=float, default=0.5)
    parser.add_argument("--models-dir", default="models", help="trained command models dir")
    parser.add_argument("--command-threshold", type=float, default=0.5)
    parser.add_argument(
        "--stt-commands",
        action="store_true",
        help="interim mode: resolve commands via Whisper + closed-set match "
        "(until trained command models exist)",
    )
    parser.add_argument("--stt-model", default="base.en", help="faster-whisper model size")
    parser.add_argument("--arm-window", type=float, default=4.0, help="seconds to wait for a command after wake")
    parser.add_argument("--cooldown", type=float, default=2.0, help="seconds ignored after dispatch (debounce)")
    parser.add_argument("--lat", type=float, default=0.0, help="marker latitude")
    parser.add_argument("--lon", type=float, default=0.0, help="marker longitude")
    parser.add_argument("--cot-url", default=DEFAULT_COT_URL)
    parser.add_argument("--local-addr", default=None, help="NIC IP to pin multicast egress to")
    parser.add_argument("--dry-run", action="store_true", help="print CoT events instead of sending")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(list_input_devices())
        return

    if args.wake_model is None:
        custom = Path(args.models_dir) / f"{model_stem(TARGET_WAKE_PHRASE)}.onnx"
        args.wake_model = str(custom) if custom.exists() else STAND_IN_MODEL
        if args.wake_model == STAND_IN_MODEL:
            print(
                f'[voice] no {custom} yet — using "{STAND_IN_MODEL}" as stand-in '
                f'wake phrase (say "hey jarvis"); see docs/training.md',
                flush=True,
            )

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\n[voice] stopped")


if __name__ == "__main__":
    main()
