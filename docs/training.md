# Training the wake and command models

The pipeline needs one openwakeword model per phrase, dropped into `models/`
(gitignored — deliberately un-ignore them in `.gitignore` if you want them
versioned). Filenames follow the phrase-with-underscores convention that
`tak_vcp.voice.classifier` resolves:

| Phrase | File |
|---|---|
| activate tak (wake, layer 1) | `models/activate_tak.onnx` |
| mark contact | `models/mark_contact.onnx` |
| mark friendly | `models/mark_friendly.onnx` |
| mark casualty | `models/mark_casualty.onnx` |
| drop waypoint | `models/drop_waypoint.onnx` |

Until `activate_tak.onnx` exists, `tak-vcp-voice` falls back to the pretrained
`hey_jarvis` stand-in for the wake layer; until command models exist,
`--stt-commands` (Whisper + closed-set snap) is the interim command mode.

## How to train

**Local pipeline (this repo): [training/README.md](../training/README.md).**
openwakeword's own trainer with the Piper TTS stage replaced by Edge neural
TTS (Piper's generator doesn't run on Windows). Configs are generated from the
command vocabulary, so `training/phrases.py` + `tak_vcp/commands.py` stay the
single source of truth.

Alternative: openwakeword's `notebooks/automatic_model_training.ipynb`
([openWakeWord repo](https://github.com/dscripka/openWakeWord)) on Colab's free
GPU tier — same result, Piper included; download the `.onnx` into `models/`
under the names above.

Training signal that matters either way:

- **Noise augmentation**: mix in the actual operating noise profile — vehicle,
  wind, radio chatter, raised voices. Locally: put those recordings in
  `training/data/background/`.
- **Real recordings**: synthetic TTS bootstraps fine, but commands issued
  under stress (raised voice, breathing hard, clipped delivery) don't sound
  like TTS. Record real operators across the phrase set, add them to the
  positive clips, retrain before trusting the models operationally.

## Validation before trusting a model set

1. **Smoke test**: stream a TTS clip of each phrase through each model
   (16 kHz mono, 1280-sample frames — see `WakeWordListener.score`). Expect
   high (>0.5) on its own phrase, near zero elsewhere. The pretrained
   `hey_jarvis` benchmark on this machine: 0.998 own-phrase, 0.000 others.
2. **Cross-activation matrix**: every model vs. recordings of every *other*
   phrase. Small classifier heads bleed on similar phrases — `mark contact`
   vs `mark casualty` share the "mark c…" onset and need explicit testing.
   The generated configs already train each command against the others as
   adversarial negatives; the matrix verifies it worked.
3. **Wake phrase false-accepts**: probe "activate tak" against
   near-collisions — "activate", "attack", "tak" alone, "activate that"
   (these are in `training/phrases.py:WAKE_ADVERSARIAL`). The runtime supports
   multiple wake models side by side (`--wake-model a.onnx b.onnx`), so if one
   phrase proves flaky in the field, train an alternate ("TAK online",
   "TAK wake up") and run both.
4. **Thresholds**: tune per-command (`--wake-threshold`, `--command-threshold`
   are global today; per-command thresholds are an open item) on the
   false-accept/false-reject tradeoff — affiliation-setting commands should
   err toward false-reject.
