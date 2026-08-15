# Local training pipeline

Trains the `activate tak` wake model and one model per command phrase using
openwakeword's own trainer, with the Piper TTS stage (no Windows support)
replaced by Edge neural TTS. Everything runs from the **repo root**.

```sh
uv sync --extra voice --group train

# 1. Write per-model YAML configs (regenerate after vocabulary changes)
uv run python training/make_configs.py

# 2. Synthesize positive + adversarial-negative clips (network: Edge TTS)
uv run python training/generate_samples.py --all --n 400

# 3. Fetch RIRs + precomputed negative features (~6 GB, one-off);
#    then put REAL operating-environment noise wavs in training/data/background/
uv run python training/download_data.py --placeholder-noise

# 4. Augment + train (repeat per config, or loop them)
uv run python -m openwakeword.train --training_config training/config/activate_tak.yml --augment_clips --train_model

# 5. Deploy: copy the trained .onnx into models/ — the runtime picks it up
#    automatically (tak-vcp-voice switches off the hey_jarvis stand-in /
#    classifier mode engages once command models exist)
copy training\output\activate_tak\activate_tak.onnx models\
```

Notes:

- `training/phrases.py` is the source of truth for what gets trained; command
  phrases come straight from `tak_vcp.commands`. Each command's adversarial
  negatives are the *other* commands (cross-activation defense); the wake
  phrase's are near-collisions ("attack", "activate", ...).
- `piper_stub/` satisfies train.py's unconditional Piper import — never call
  `--generate_clips`; step 2 above replaces it.
- Synthetic TTS bootstraps the models. Before trusting them operationally,
  record real operators (raised voice, breathing hard, clipped delivery), add
  the clips to `positive_train`/`positive_test`, and retrain.
- Validation checklist (cross-activation matrix, false-accept probes,
  thresholds): see docs/training.md.
- `training/data/` and `training/output/` are gitignored (gigabytes).
