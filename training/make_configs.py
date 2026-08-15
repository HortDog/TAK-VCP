"""Write one openwakeword training YAML per model into training/config/.

Values follow openwakeword's automatic-training notebook template; the ones
worth tuning are n_samples, steps, and max_negative_weight. Paths are relative
to the repo root — run everything from there.

    uv run python training/make_configs.py
"""

from pathlib import Path

import yaml

from phrases import model_specs

TRAINING = Path(__file__).resolve().parent
CONFIG_DIR = TRAINING / "config"


def config_for(spec: dict) -> dict:
    return {
        "model_name": spec["name"],
        "target_phrase": [spec["phrase"]],
        "custom_negative_phrases": spec["negatives"],
        "model_type": "dnn",
        "layer_size": 32,
        "n_samples": 2000,
        "n_samples_val": 400,
        "tts_batch_size": 50,  # unused (piper stage is stubbed)
        "piper_sample_generator_path": "training/piper_stub",
        "output_dir": "training/output",
        "rir_paths": ["training/data/rirs"],
        "background_paths": ["training/data/background"],
        "background_paths_duplication_rate": [1],
        "augmentation_rounds": 1,
        "augmentation_batch_size": 16,
        "feature_data_files": {
            "ACAV100M_sample": "training/data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        },
        "false_positive_validation_data_path": "training/data/validation_set_features.npy",
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
        "steps": 20000,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "total_length": 32000,  # 2 s clips at 16 kHz
    }


def main() -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    for spec in model_specs():
        path = CONFIG_DIR / f"{spec['name']}.yml"
        path.write_text(yaml.dump(config_for(spec), sort_keys=False), encoding="utf-8")
        print(f"wrote {path.relative_to(TRAINING.parent)}")


if __name__ == "__main__":
    main()
