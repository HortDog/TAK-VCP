# TAK-VCP — Voice-to-CoT Trigger System

Wake word + speech-to-text pipeline that fires predefined, predictable CoT
(Cursor on Target) packets to spawn markers on TAK clients. Test environment:
WinTAK + PyTAK, no plugin build yet.

Full design and rationale: [docs/voice-to-cot-handoff.md](docs/voice-to-cot-handoff.md)

```
                                    ┌─→ [bare command]  → CoT builder → PyTAK → WinTAK
Mic → Wake word (layer 1) → Command classifier (layer 2, openWakeWord)
                                    └─→ [parameterized] → Whisper (tail only) → arg parser → CoT builder → PyTAK → WinTAK
```

The command verb is resolved by a closed-set classifier (one openwakeword model
per command) and can never be anything outside the command table. Whisper is
scoped to free-text argument tails only.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.11
(openwakeword's Linux tflite-runtime dependency has no wheels beyond it).

```sh
uv sync                # core: pytak + the CoT/transport code
uv sync --extra voice  # adds openwakeword, faster-whisper, sounddevice
```

## Transport smoke test (build order step 3)

Send a hardcoded marker to WinTAK — no voice involved — to isolate
transport/formatting issues first:

```sh
uv run tak-vcp-send-test --lat 38.8895 --lon -77.0353
uv run tak-vcp-send-test --command "mark friendly" --cot-url tcp://192.168.1.50:8087
```

Default transport is UDP multicast `udp://239.2.3.1:6969`, which stock WinTAK
listens on when both machines share a LAN/VLAN. For TCP, add a matching TCP
input in WinTAK's network preferences first.

## Layout

| Path | Purpose |
|---|---|
| `src/tak_vcp/commands.py` | Closed command vocabulary → CoT type table (single source of truth) |
| `src/tak_vcp/cot.py` | CoT XML event builder |
| `src/tak_vcp/transport.py` | PyTAK worker draining the voice pipeline's queue |
| `src/tak_vcp/send_test.py` | Hardcoded one-shot send CLI (`tak-vcp-send-test`) |
| `src/tak_vcp/voice/` | Wake word / classifier / Whisper-tail layers — **stubs, not yet built** |
| `docs/voice-to-cot-handoff.md` | Design handoff doc |

## Status vs. build order

1. ✅ Repo/tooling scaffold (uv, package layout, tests)
2. ⬜ Stand up WinTAK, confirm receive path (multicast group or TCP input)
3. ⬜ `uv run tak-vcp-send-test` → marker lands on the WinTAK map
4. ⬜ Layer-1 wake word + layer-2 command classifier feeding the voice queue
5. ⬜ Whisper tail transcription + argument parser for parameterized commands
6. ⬜ Vocabulary iteration, confidence thresholds, confirmation UX

## Tests

```sh
uv run pytest
```
