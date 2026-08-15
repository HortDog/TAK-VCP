# Voice-to-CoT Trigger System — Handoff Doc

## Goal
Wake word + speech-to-text pipeline that fires predefined, predictable CoT (Cursor on Target) packets to spawn markers/objects on TAK clients. Test environment: WinTAK + PyTAK server, no plugin build yet.

## Pipeline overview

Hybrid two-layer design, split by whether the command takes arguments:

```
                                    ┌─→ [bare command]  → CoT builder → PyTAK → WinTAK
Mic → Wake word (layer 1) → Command classifier (layer 2, openWakeWord)
                                    └─→ [parameterized] → Whisper (tail only) → arg parser → CoT builder → PyTAK → WinTAK
```

Design principle: use a **small, fixed command vocabulary** resolved by a closed-set classifier, not open-vocabulary transcription. Whisper stays in the pipeline only for free-text arguments, where a misrecognition is cosmetic rather than tactical. See [Design decision: layer-2 wake models vs Whisper transcription](#design-decision-layer-2-wake-models-vs-whisper-transcription) for the rationale.

## Components

### 1. Wake word (layer 1)
- Library: `openwakeword` (runs locally, low CPU, offline-capable).
- Continuously listens for the activation phrase; on trigger, arms the command layer.

### 2. Command classifier (layer 2)
- One trained `openwakeword` model per command in the vocabulary.
- Closed set: cannot emit anything outside the command table. This is the key safety property.
- Fires on streaming frames (~80 ms), so a bare command dispatches without any STT pass in the path.
- Per-command confidence scores come out directly — usable for thresholding and for deciding when to demand confirmation.
- Phonetic distinctness still matters. Small classifier heads trained on similar phrases cross-activate; `mark contact` vs `mark contact lost` will bleed. Keep commands acoustically far apart.

### 3. Speech-to-text (arguments only)
- Library: `faster-whisper` (or base `whisper`).
- Invoked **only** for parameterized commands, transcribing the tail of the utterance after the classifier has already resolved the action.
- Example: `mark contact bearing zero four five range two hundred` — classifier resolves `mark contact` (affiliation/type, safety-relevant); Whisper handles the bearing/range tail.
- Add a confirmation step (tone or repeat-back) before dispatching, especially for anything that sets affiliation/type.

### 4. CoT event builder
Maps a recognized command to a CoT `type` and generates the XML event.

```python
from datetime import datetime, timedelta, timezone

def build_marker_cot(command: str, lat: float, lon: float) -> bytes:
    type_map = {
        "mark contact": "a-h-G",       # hostile ground
        "mark friendly": "a-f-G",      # friendly ground
        "mark casualty": "a-f-G-U-C",  # casualty
        "drop waypoint": "b-m-p-w",    # waypoint
    }
    cot_type = type_map.get(command, "b-m-p-w")
    now = datetime.now(timezone.utc)
    stale = now + timedelta(minutes=5)
    uid = f"voice-{int(now.timestamp())}"

    xml = f"""<event version="2.0" uid="{uid}" type="{cot_type}" how="m-g"
        time="{now.isoformat()}" start="{now.isoformat()}" stale="{stale.isoformat()}">
        <point lat="{lat}" lon="{lon}" hae="0" ce="10" le="10"/>
        <detail>
            <contact callsign="{uid}"/>
        </detail>
    </event>"""
    return xml.encode()
```

Notes:
- `how="m-g"` = manually generated.
- `stale` window (5 min here) controls how long the marker persists before TAK considers it expired — tune per use case.
- UID derived from timestamp to avoid collisions; revisit if wake word can double-fire (debounce recommended).

### 5. PyTAK transport
PyTAK handles serialization and delivery (TCP, UDP multicast, or to a TAK Server) so the voice pipeline doesn't need raw socket handling.

```python
import asyncio
import pytak
from configparser import ConfigParser

class VoiceCommandWorker(pytak.QueueWorker):
    async def handle_data(self, cot_event):
        await self.put_queue(cot_event)

    async def run(self):
        while True:
            # pulled from the voice pipeline's output queue
            cot_xml = await self.voice_queue.get()
            await self.handle_data(cot_xml)

async def main():
    config = ConfigParser()
    config["mycottool"] = {"COT_URL": "udp://239.2.3.1:6969"}  # or tcp://<wintak_ip>:8087
    clitool = pytak.CLITool(config["mycottool"])
    await clitool.setup()
    clitool.add_tasks({VoiceCommandWorker(clitool.tx_queue, config["mycottool"])})
    await clitool.run()

asyncio.run(main())
```

Transport options:
- **Multicast** (`udp://239.2.3.1:6969`) — default WinTAK listens on this if both machines share a LAN/VLAN. No extra config needed on WinTAK side.
- **TCP point-to-point** (`tcp://<WinTAK-host-ip>:8087`) — requires adding a matching TCP input in WinTAK's network preferences.
- **Via TAK Server** — point `COT_URL` at the server; WinTAK subscribes through its normal server connection.

## Design decision: layer-2 wake models vs Whisper transcription

**Question:** rely on phonetically distinct phrases transcribed by Whisper, or train a second layer of openWakeWord models as a command classifier?

**Decision: trained command models for the verb, Whisper scoped to arguments only.**

### Why the classifier wins for bare commands

- **Closed set vs open vocabulary.** A command classifier physically cannot emit something outside the command table. Whisper can — and its characteristic failure mode is the wrong one for this pipeline: on noise, silence, or clipped audio it doesn't return low confidence, it returns confident garbage, often hallucinated stock phrases from training data. In a system that auto-dispatches CoT, "confidently wrong" is worse than "unsure."
- **Latency.** openWakeWord command models fire on streaming ~80 ms frames the instant the phrase completes. Whisper needs VAD to detect end-of-utterance, then a transcription pass. Tens of ms vs hundreds-to-thousands.
- **Marginal cost per command is low.** openWakeWord models share a common front end (melspectrogram + speech embedding model); each wake word is a small classifier head on top. Adding command #8 doesn't cost 8× the compute — the shared embedding dominates. Scales better than "N models running" implies.
- **Direct per-command confidence.** You get a score per command model (`mark contact 0.91 / mark friendly 0.34`), immediately usable for thresholds and confirmation logic. Whisper gives text; mapping that to per-command confidence is indirect.
- **Environment-tunable.** The training pipeline uses synthetic TTS samples with noise augmentation — augment with the actual operating noise profile (vehicle, wind, radio chatter, raised voice). Can't do that to Whisper without fine-tuning, a far bigger job.
- **Portability to the plugin endgame.** If this becomes an ATAK plugin on Android, tflite classifier heads port cleanly to the Android TFLite runtime. Whisper on a handset is whisper.cpp/ONNX territory — doable but much heavier.

### Where the classifier breaks down

Arguments. A classifier handles `mark contact` perfectly and cannot handle `mark contact bearing zero four five, range two hundred`, free-text labels, or callsigns. The moment a command takes parameters, transcription is required. Hence the hybrid split.

### Optional: cross-check mode for affiliation-setting commands

For commands that put a hostile marker on a shared map, consider running both paths and voting:
- **Agreement** → dispatch silently, high confidence.
- **Disagreement or single-fire** → require confirmation.

This forfeits the latency and compute savings (Whisper runs anyway), but a two-source vote is cheap insurance for the highest-consequence commands. Variant that keeps the latency win: dispatch optimistically on the classifier, then flag or retract on Whisper mismatch.

### Training data caveat

The automated pipeline generates synthetic TTS samples, which is fine for a wake word said calmly. Commands issued under stress — raised voice, breathing hard, clipped delivery — don't sound like Piper output. Budget for recording real samples from actual operators across the command set, even if bootstrapping with synthetic.

## Recommended build/test order

1. **Stand up WinTAK**, confirm it's receiving on the chosen multicast group or TCP port (check Network Manager / inputs panel).
2. **Run PyTAK script** pointed at that address.
3. **Send a hardcoded CoT event** (skip voice entirely) to confirm a marker lands on the WinTAK map. This isolates transport/formatting issues before adding audio complexity.
4. **Wire in layer-1 wake word + layer-2 command classifier**, feeding recognized bare commands into `voice_queue`. No Whisper in the loop at this stage.
5. **Add Whisper** only once bare commands dispatch reliably, scoped to parameterized command tails.
6. Iterate on command vocabulary and confirmation UX once markers are landing reliably.

## Open items / not yet built
- Layer-1 wake word + layer-2 command classifier listener loop (feeds `voice_queue`).
- Training set for the command models — synthetic TTS bootstrap, then real operator recordings under stress conditions.
- Cross-activation testing across the command vocabulary; prune or rename phrases that bleed.
- Whisper tail-transcription path and argument parser (bearing/range, labels, callsigns).
- Confidence thresholds per command, and confirmation-trigger policy.
- WinTAK-side network input configuration steps (TCP inputs, multicast verification).
- Debounce/dedup logic for repeated wake-word triggers.
- Eventually: native ATAK plugin path (Kotlin/Java, TAK SDK `MapItem`/`CotMapComponent`) if moving beyond PyTAK bridge testing — heavier but more integrated (works offline, no external process).

## Reference type codes used above
| Command         | CoT type      | Meaning              |
|------------------|---------------|-----------------------|
| mark contact     | `a-h-G`       | Hostile, ground       |
| mark friendly    | `a-f-G`       | Friendly, ground      |
| mark casualty    | `a-f-G-U-C`   | Casualty, friendly    |
| drop waypoint    | `b-m-p-w`     | Generic waypoint      |
