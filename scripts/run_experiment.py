"""
Run a full experiment end-to-end: feature extraction → training → evaluation.

Usage:
    python scripts/run_experiment.py configs/experiments/bandpower_rf_v1.yaml
    python scripts/run_experiment.py configs/experiments/bandpower_rf_v1.yaml --skip-extraction
"""

import sys
import argparse
import subprocess
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(cmd: list):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment config YAML")
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip feature extraction (use if features already exist)"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training (dry run to check configs only)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    if config.get("description"):
        print(f"  {config['description'].strip()}")
    print(f"{'='*60}\n")

    # Step 1: Feature extraction
    if not args.skip_extraction:
        print("STEP 1: Extracting features...")
        run([sys.executable, "scripts/extract_features.py", config["features_config"]])
    else:
        print("STEP 1: Skipped (--skip-extraction)")

    # Step 2: Training + CV evaluation
    if not args.skip_training:
        print("\nSTEP 2: Training and evaluating...")
        train_cmd = [sys.executable, "scripts/train.py", config["training_config"]]
        if config.get("save_final_model", False):
            train_cmd.append("--save-model")
        run(train_cmd)
    else:
        print("STEP 2: Skipped (--skip-training)")

    print(f"\n{'='*60}")
    print(f"  Experiment '{name}' complete.")
    print(f"  Results: outputs/results/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()