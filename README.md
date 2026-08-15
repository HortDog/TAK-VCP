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

Transport notes (learned the hard way, WinTAK 4.1 on Windows 11):

- **Same host as WinTAK: use `--cot-url tcp://127.0.0.1:8087`.** Stock WinTAK
  already listens on TCP 4242 and 8087 — no input config needed. This is the
  verified path (marker confirmed on the map).
- **Multicast (`udp+wo://239.2.3.1:6969`, the default) is for cross-host
  LAN.** On the same host it does not deliver: WinTAK joins the group and the
  firewall allows it, but loopback multicast never rendered a marker in
  testing.
- The `+wo` (write-only) modifier is required whenever WinTAK runs on the same
  host — without it pytak also binds a receive socket on the port and collides
  with WinTAK's exclusive bind (WinError 10013).
- On multi-homed hosts (e.g. a VPN adapter present), pass
  `--local-addr <lan-ip>` to pin which NIC the multicast egresses on.

If a marker doesn't appear, run `uv run tak-vcp-listen` (stop WinTAK first —
it holds 6969 exclusively — or use a different `--port`) to check whether
packets are on the wire at all.

## Layout

| Path | Purpose |
|---|---|
| `src/tak_vcp/commands.py` | Closed command vocabulary → CoT type table (single source of truth) |
| `src/tak_vcp/cot.py` | CoT XML event builder |
| `src/tak_vcp/transport.py` | PyTAK worker draining the voice pipeline's queue |
| `src/tak_vcp/send_test.py` | Hardcoded one-shot send CLI (`tak-vcp-send-test`) |
| `src/tak_vcp/recv_test.py` | Multicast receive debug CLI (`tak-vcp-listen`) — is anything on the wire? |
| `src/tak_vcp/voice/` | Wake word / classifier / Whisper-tail layers — **stubs, not yet built** |
| `docs/voice-to-cot-handoff.md` | Design handoff doc |

## Status vs. build order

1. ✅ Repo/tooling scaffold (uv, package layout, tests)
2. ✅ WinTAK receive path confirmed — default TCP inputs on 4242/8087; wire
   verified with `tak-vcp-listen` on multicast
3. ✅ `uv run tak-vcp-send-test --cot-url tcp://127.0.0.1:8087` → marker
   confirmed on the WinTAK map (2026-08-16, same-host)
4. ⬜ Layer-1 wake word + layer-2 command classifier feeding the voice queue
5. ⬜ Whisper tail transcription + argument parser for parameterized commands
6. ⬜ Vocabulary iteration, confidence thresholds, confirmation UX

## Tests

```sh
uv run pytest
```
