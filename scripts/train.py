"""
Train and evaluate a model on pre-extracted features.

Usage:
    python scripts/train.py configs/training/bandpower_rf.yaml
    python scripts/train.py configs/training/bandpower_rf_thresh.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.cross_val import (
    cross_val_bandpower_rf,
    cross_val_stft_cnn,
    cross_val_connectivity,
    cross_val_raw_cnn,
    train_final_bandpower_rf,
)
from src.training.threshold import cross_val_bandpower_rf_threshold


TRAIN_FN = {
    "bandpower_rf":       cross_val_bandpower_rf,
    "bandpower_rf_thresh": cross_val_bandpower_rf_threshold,
    "stft_cnn":           cross_val_stft_cnn,
    "connectivity":       cross_val_connectivity,
    "raw_cnn":            cross_val_raw_cnn,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to training config YAML")
    parser.add_argument("--save-model", action="store_true",
                        help="Also train and save a final model on full data (RF only)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mode = config["mode"]
    version = config.get("version", "v1")
    output_dir = f"outputs/results/{mode}_{version}"

    fn = TRAIN_FN.get(mode)
    if fn is None:
        raise ValueError(f"Unknown training mode '{mode}'. Choose from: {list(TRAIN_FN)}")

    features_dir = config["features_dir"]
    subjects = config.get("subjects", list(range(1, 66)))

    # Pass everything except meta-keys as kwargs
    skip = {"mode", "version", "features_dir", "subjects"}
    kwargs = {k: v for k, v in config.items() if k not in skip}

    print(f"\nTraining [{mode}] on {features_dir}  →  {output_dir}\n")
    results = fn(features_dir=features_dir, subjects=subjects, output_dir=output_dir, **kwargs)

    print("\nResults:")
    for k, v in results.items():
        if not isinstance(v, object.__class__):  # skip DataFrames
            print(f"  {k}: {v}")

    if args.save_model and mode in ("bandpower_rf", "bandpower_rf_thresh"):
        model_dir = f"outputs/models/{mode}_{version}"
        train_final_bandpower_rf(
            features_dir=features_dir,
            subjects=subjects,
            rf_params=config.get("rf_params", {}),
            channels=config.get("channels"),
            output_dir=model_dir,
        )


if __name__ == "__main__":
    main()