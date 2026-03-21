"""
Run inference on a single raw EEG .set file using the trained bandpower RF model.

Usage:
    python scripts/predict.py --subject 1
    python scripts/predict.py --set-file path/to/recording.set
    python scripts/predict.py --set-file path/to/recording.set --threshold 0.362

The pipeline:
    1. Load raw .set file
    2. Bandpass filter + average reference
    3. Extract 2s sliding windows (0.5s step)
    4. Compute bandpower (Welch) for 5 electrodes
    5. Aggregate window-level predictions → subject-level decision
    6. Output prediction, confidence, and per-window breakdown

Note: The saved model was trained on subjects 1-65 from ds004504.
      For genuine unseen inference, use recordings from a different dataset.
"""

import sys
import argparse
import numpy as np
import torch
import joblib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_raw_eeg, get_diagnosis
from src.data.transforms import preprocess_raw, extract_raw_windows
from src.features.bandpower import compute_bandpower

# ---------------------------------------------------------------------------
# Defaults — match exactly what the model was trained on
# ---------------------------------------------------------------------------
MODEL_PATH   = Path("outputs/models/bandpower_rf_thresh_v1/model.joblib")
CHANNELS     = [12, 14, 9, 16, 10]   # T4, Pz, Cz, T6, C3
THRESHOLD    = 0.362                  # mean optimized threshold from CV
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4", "T5", "P3",
    "Pz",  "P4",  "T6", "O1", "O2",
]
PREPROCESS_CONFIG = {
    "filters":            [1, 40],
    "window_size":        2.0,
    "step_size":          0.5,
    "artifact_rejection": True,
    "ptp_threshold":      100e-6,
}


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def predict_from_raw(raw, model, threshold: float) -> dict:
    """
    Run the full feature extraction + inference pipeline on an MNE Raw object.

    Returns a results dict with:
        prediction:       "AD" or "HC"
        ad_probability:   float, mean probability of AD across windows
        confidence:       "High" / "Medium" / "Low"
        above_threshold:  fraction of windows predicting AD
        n_windows:        number of windows processed
        window_probs:     np.ndarray of per-window AD probabilities
    """
    # Step 1: build a minimal subject_data dict (no demographics needed)
    subject_data = {"raw": raw}

    # Step 2: preprocess
    raw_filtered = preprocess_raw(subject_data, PREPROCESS_CONFIG)

    # Step 3: window
    windows_data = extract_raw_windows(raw_filtered, PREPROCESS_CONFIG, subject_id="?")

    if len(windows_data["windows"]) == 0:
        raise ValueError("No windows extracted — recording may be too short or all windows rejected.")

    # Step 4: bandpower for selected channels only
    bp_tensor, _ = compute_bandpower(windows_data, sfreq=windows_data["sfreq"])
    # bp_tensor: (n_windows, 19, 5) — select the 5 channels
    bp_selected = bp_tensor[:, CHANNELS, :]          # (n_windows, 5, 5)
    X = bp_selected.reshape(len(bp_selected), -1).numpy()   # (n_windows, 25)

    # Step 5: per-window probabilities
    window_probs = model.predict_proba(X)[:, 0]  # col 0 = AD (label=0)

    # Step 6: aggregate — mean probability across windows
    mean_prob = float(np.mean(window_probs))
    fraction_above = float(np.mean(window_probs >= threshold))

    prediction = "AD" if mean_prob >= threshold else "HC"

    # Confidence: distance from threshold
    distance = abs(mean_prob - threshold)
    if distance >= 0.15:
        confidence = "High"
    elif distance >= 0.07:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "prediction":      prediction,
        "ad_probability":  round(mean_prob, 4),
        "threshold":       threshold,
        "confidence":      confidence,
        "above_threshold": round(fraction_above, 4),
        "n_windows":       len(window_probs),
        "window_probs":    window_probs,
    }


def print_report(results: dict, true_label: str = None):
    """Print a clean inference report to stdout."""
    prob_pct = results["ad_probability"] * 100
    bar_len  = 40
    filled   = int(bar_len * results["ad_probability"])
    bar      = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "=" * 55)
    print("  EEG ALZHEIMER'S SCREENING — INFERENCE REPORT")
    print("=" * 55)
    print(f"  Prediction:      {results['prediction']}  ({results['confidence']} confidence)")
    print(f"  AD probability:  {prob_pct:.1f}%")
    print(f"  [{bar}]")
    print(f"  Threshold:       {results['threshold']:.3f}  "
          f"({results['above_threshold']*100:.1f}% of windows above)")
    print(f"  Windows used:    {results['n_windows']}")

    if true_label is not None:
        correct = "✓ CORRECT" if results["prediction"] == true_label else "✗ INCORRECT"
        print(f"  True label:      {true_label}  {correct}")

    print("=" * 55)

    # Per-window histogram (ASCII, 10 bins)
    probs = results["window_probs"]
    counts, edges = np.histogram(probs, bins=10, range=(0, 1))
    print("\n  Per-window AD probability distribution:")
    max_count = max(counts) if max(counts) > 0 else 1
    for i, (count, edge) in enumerate(zip(counts, edges[:-1])):
        bar = "▪" * int(count / max_count * 20)
        marker = " ← threshold" if edges[i] <= results["threshold"] < edges[i+1] else ""
        print(f"  {edge:.1f}-{edges[i+1]:.1f}  {bar:<20} {count:>4}{marker}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run Alzheimer's screening inference on a raw EEG .set file."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--subject", type=int,
        help="Subject ID from ds004504 (1-65). Loads from data/ds004504/."
    )
    source.add_argument(
        "--set-file", type=str,
        help="Path to a raw .set file from any source."
    )

    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Decision threshold (default: {THRESHOLD}, optimized for sensitivity)"
    )
    parser.add_argument(
        "--model", type=str, default=str(MODEL_PATH),
        help="Path to trained .joblib model file"
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Optionally save results as JSON to this path"
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Data root directory (used with --subject)"
    )

    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python scripts/train.py configs/training/bandpower_rf_thresh.yaml --save-model")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load EEG
    true_label = None

    if args.subject is not None:
        print(f"Loading subject {args.subject} from ds004504...")
        subject_data = load_raw_eeg(args.subject, data_root=args.data_root)
        if subject_data is None:
            print(f"ERROR: Could not load subject {args.subject}")
            sys.exit(1)
        raw = subject_data["raw"]
        true_label = subject_data["diagnosis"]
        print(f"  True diagnosis (known): {true_label}")
        print(f"  NOTE: This subject was in the training set. "
              f"Use --set-file with an external dataset for genuine inference.")

    else:
        set_path = Path(args.set_file)
        if not set_path.exists():
            print(f"ERROR: .set file not found: {set_path}")
            sys.exit(1)
        print(f"Loading {set_path}...")
        import mne
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        except Exception as e:
            print(f"ERROR loading .set file: {e}")
            sys.exit(1)

    # Run inference
    print(f"Running inference ({len(CHANNELS)} electrodes: "
          f"{[CHANNEL_NAMES[c] for c in CHANNELS]})...")
    results = predict_from_raw(raw, model, threshold=args.threshold)

    # Report
    print_report(results, true_label=true_label)

    # Optionally save JSON (exclude window_probs array for readability)
    if args.save_json:
        save_data = {k: v for k, v in results.items() if k != "window_probs"}
        if true_label:
            save_data["true_label"] = true_label
            save_data["correct"] = results["prediction"] == true_label
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()