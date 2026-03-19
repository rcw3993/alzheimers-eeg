"""
Extract features for all subjects and save as .pt files.

Usage:
    python scripts/extract_features.py configs/features/bandpower.yaml
    python scripts/extract_features.py configs/features/bandpower.yaml --subjects 1 2 3
"""

import sys
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure project root is on path when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_raw_eeg, get_diagnosis
from src.data.transforms import preprocess_raw, extract_raw_windows
from src.features.bandpower import compute_bandpower
from src.features.stft import compute_stft
from src.features.connectivity import compute_plv
from src.features.raw import compute_raw


REPRESENTATION_FN = {
    "bandpower":    compute_bandpower,
    "stft":         compute_stft,
    "connectivity": compute_plv,
    "raw":          compute_raw,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to feature config YAML")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Override subject list from config")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-base", default="outputs/features")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rep = config["representation"]
    version = config.get("version", "v1")
    output_dir = Path(args.output_base) / f"{rep}_{version}"
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    subjects = args.subjects or config.get("subjects", list(range(1, 66)))
    diag_filter = config.get("diagnosis_filter", None)
    if diag_filter:
        subjects = [s for s in subjects if get_diagnosis(s) in diag_filter]

    print(f"\nExtracting [{rep}] for {len(subjects)} subjects → {output_dir}\n")

    compute_fn = REPRESENTATION_FN[rep]
    summary_rows = []

    for subject_id in subjects:
        subject_data = load_raw_eeg(subject_id, data_root=args.data_root)
        if subject_data is None:
            continue

        raw_filtered = preprocess_raw(subject_data, config)
        windows_data = extract_raw_windows(raw_filtered, config, subject_id)
        sfreq = windows_data["sfreq"]

        # All feature functions accept (windows_data, sfreq, **extra_kwargs)
        extra = {k: config[k] for k in config if k not in
                 ("representation", "version", "subjects", "filters",
                  "window_size", "step_size", "artifact_rejection",
                  "ptp_threshold", "diagnosis_filter", "name")}

        result = compute_fn(windows_data, sfreq=sfreq, **extra)

        # Unpack — raw returns just tensor; others return (tensor, metadata)
        if isinstance(result, tuple):
            tensor = result[0]
        else:
            tensor = result

        out_path = output_dir / "data" / f"sub-{subject_id:03d}_{rep}.pt"
        torch.save(tensor, out_path)

        summary_rows.append({
            "subject_id":  subject_id,
            "diagnosis":   subject_data["diagnosis"],
            "n_windows":   tensor.shape[0],
            "tensor_shape": str(tuple(tensor.shape)),
            "age":         subject_data["demographics"]["age"],
            "gender":      subject_data["demographics"]["gender"],
        })
        print(f"  sub-{subject_id:03d}: {tuple(tensor.shape)} → {out_path.name}")

    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "tables" / "summary.csv", index=False)
    print(f"\nDone. {len(summary_rows)} subjects saved to {output_dir}")
    print(df[["subject_id", "diagnosis", "n_windows", "tensor_shape"]])


if __name__ == "__main__":
    main()