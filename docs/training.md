# Training the wake and command models

The pipeline needs one openwakeword model per phrase, dropped into `models/`
(gitignored — deliberately un-ignore them in `.gitignore` if you want them
versioned). Filenames follow the phrase-with-underscores convention that
`tak_vcp.voice.classifier` resolves:

| Phrase | File |
|---|---|
| tak active (wake, layer 1) | `models/tak_active.onnx` |
| mark contact | `models/mark_contact.onnx` |
| mark friendly | `models/mark_friendly.onnx` |
| mark casualty | `models/mark_casualty.onnx` |
| drop waypoint | `models/drop_waypoint.onnx` |

Until `tak_active.onnx` exists, `tak-vcp-voice` falls back to the pretrained
`hey_jarvis` stand-in for the wake layer; until command models exist, use
`--stt-commands` (Whisper + closed-set snap, interim only).

## How to train

Use openwakeword's automated pipeline: `notebooks/automatic_model_training.ipynb`
in the [openWakeWord repo](https://github.com/dscripka/openWakeWord) (designed
to run on Colab's free GPU tier). Per phrase: enter the target phrase, let it
generate synthetic TTS samples (Piper) with augmentation, train, and download
the resulting `.onnx` into `models/` under the name above.

Training knobs that matter for this project:

- **Noise augmentation**: mix in the actual operating noise profile — vehicle,
  wind, radio chatter, raised voices — not just the default noise sets.
- **Real recordings**: synthetic TTS is fine to bootstrap, but commands issued
  under stress (raised voice, breathing hard, clipped delivery) don't sound
  like Piper output. Record real operators across the command set and include
  them before trusting the models operationally.

## Validation before trusting a model set

1. **Smoke test**: stream a TTS clip of each phrase through each model
   (16 kHz mono, 1280-sample frames — see `WakeWordListener.score`). Expect
   high (>0.5) on its own phrase, near zero elsewhere. The pretrained
   `hey_jarvis` benchmark on this machine: 0.998 own-phrase, 0.000 others.
2. **Cross-activation matrix**: every model vs. recordings of every *other*
   phrase. Small classifier heads bleed on similar phrases — `mark contact`
   vs `mark casualty` share the "mark c…" onset and need explicit testing.
   Prune or rename phrases that cross-fire.
3. **Wake phrase false-accepts**: "tak active" repeats the /æk/ syllable
   (tak-ACK-tive). Test against near-collisions — "attack", "active",
   "tactic", "TAK" alone. If it bleeds, candidate replacements that keep the
   spirit: "activate TAK", "TAK online", "TAK wake up" — retrain and re-run
   the matrix.
4. **Thresholds**: tune per-command (`--wake-threshold`, `--command-threshold`
   are global today; per-command thresholds are an open item) on the
   false-accept/false-reject tradeoff — affiliation-setting commands should
   err toward false-reject.
