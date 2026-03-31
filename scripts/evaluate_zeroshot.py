"""
Zero-shot cross-condition evaluation: train on ds004504 (eyes-closed),
evaluate on ds006036 (eyes-open photic stimulation).

Same 88 subjects, same electrodes, same diagnosis mapping.
No fine-tuning — raw transfer to test cross-condition generalization.

Usage:
    python scripts/evaluate_zeroshot.py
    python scripts/evaluate_zeroshot.py --subjects 1 2 3   # quick test

Reads:
    outputs/models/bandpower_rf_thresh_v1/model.joblib
    data/ds006036/derivatives/eeglab/sub-XXX/

Writes:
    outputs/results/zeroshot_ds006036/
        zeroshot_results.csv
        figures/
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, f1_score, accuracy_score
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_raw_eeg_006036, get_diagnosis
from src.data.transforms import preprocess_raw, extract_raw_windows
from src.features.bandpower import compute_bandpower
from src.utils import set_seed

# ---------------------------------------------------------------------------
# Config — must match training setup exactly
# ---------------------------------------------------------------------------
MODEL_PATH   = Path("outputs/models/bandpower_rf_thresh_v1/model.joblib")
CHANNELS     = [12, 14, 9, 16, 10]   # T4, Pz, Cz, T6, C3
THRESHOLD    = 0.362                  # mean CV threshold from training
OUTPUT_DIR   = Path("outputs/results/zeroshot_ds006036")
SUBJECTS     = list(range(1, 66))     # AD (1-36) + HC (37-65), skip FTD

PREPROCESS_CONFIG = {
    "filters":            [1, 40],
    "window_size":        2.0,
    "step_size":          0.5,
    "artifact_rejection": True,
    "ptp_threshold":      100e-6,
}

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4", "T5", "P3",
    "Pz",  "P4",  "T6", "O1", "O2",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE = "#4DA8DA"
RED  = "#E05C5C"
GOLD = "#F5A623"
GRAY = "#888888"


# ---------------------------------------------------------------------------
# Per-subject inference
# ---------------------------------------------------------------------------
def run_subject(subject_id: int, model, data_root: str = "data") -> dict | None:
    """
    Load ds006036 recording, extract features, return prediction dict.
    Returns None if subject file is missing.
    """
    subject_data = load_raw_eeg_006036(subject_id, data_root=data_root)
    if subject_data is None:
        return None

    raw_filtered  = preprocess_raw(subject_data, PREPROCESS_CONFIG)
    windows_data  = extract_raw_windows(raw_filtered, PREPROCESS_CONFIG, subject_id)

    if len(windows_data["windows"]) == 0:
        print(f"  Subject {subject_id}: no windows extracted, skipping")
        return None

    bp_tensor, _ = compute_bandpower(windows_data, sfreq=windows_data["sfreq"])
    bp_selected  = bp_tensor[:, CHANNELS, :]
    X            = bp_selected.reshape(len(bp_selected), -1).numpy()

    # Per-window AD probabilities
    window_probs = model.predict_proba(X)[:, 0]  # col 0 = AD
    mean_prob    = float(np.mean(window_probs))
    prediction   = "AD" if mean_prob >= THRESHOLD else "HC"
    true_label   = get_diagnosis(subject_id)

    return {
        "subject_id":      subject_id,
        "true_label":      true_label,
        "prediction":      prediction,
        "ad_probability":  round(mean_prob, 4),
        "correct":         prediction == true_label,
        "n_windows":       len(window_probs),
        "window_probs":    window_probs,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_roc(results_df, all_probs, all_true, save_path):
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    roc_auc = auc(fpr, tpr)

    # Compare to training AUC for reference
    train_auc = 0.815

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=RED, linewidth=2.5,
            label=f"Zero-shot ROC (AUC = {roc_auc:.3f})")
    ax.axhline(0, color="none")  # padding
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1,
            label="Chance")
    ax.axvline(1 - results_df["specificity"].mean(), color=GOLD,
               linestyle=":", linewidth=1.5,
               label=f"Operating point (thr={THRESHOLD:.3f})")

    # Annotate degradation vs training
    ax.text(0.55, 0.15,
            f"Training AUC (eyes-closed): {train_auc:.3f}\n"
            f"Zero-shot AUC (eyes-open):  {roc_auc:.3f}\n"
            f"Degradation: {train_auc - roc_auc:+.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Zero-Shot ROC — ds006036 (Eyes-Open)\n"
                 "Model trained on ds004504 (Eyes-Closed), no fine-tuning")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probability_distribution(results_df, save_path):
    """Per-subject AD probability distributions for AD vs HC subjects."""
    ad_probs = results_df[results_df["true_label"] == "AD"]["ad_probability"]
    hc_probs = results_df[results_df["true_label"] == "HC"]["ad_probability"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ad_probs, bins=15, alpha=0.7, color=RED,
            label=f"AD subjects (n={len(ad_probs)})", edgecolor="darkred")
    ax.hist(hc_probs, bins=15, alpha=0.7, color=BLUE,
            label=f"HC subjects (n={len(hc_probs)})", edgecolor="navy")
    ax.axvline(THRESHOLD, color=GOLD, linestyle="--", linewidth=2,
               label=f"Decision threshold ({THRESHOLD:.3f})")
    ax.set_xlabel("Mean AD Probability (per subject)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Zero-Shot Subject-Level AD Probability Distribution\n"
                 "ds006036 (Eyes-Open) — Model trained on ds004504 (Eyes-Closed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_subject_heatmap(results_df, save_path):
    """Per-subject prediction grid showing correct/incorrect calls."""
    n = len(results_df)
    cols = 10
    rows = (n + cols - 1) // cols

    fig, ax = plt.subplots(figsize=(14, rows * 1.2 + 1))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")

    for i, row in enumerate(results_df.itertuples()):
        col_idx = i % cols
        row_idx = rows - 1 - (i // cols)

        # Color: green=correct, red=incorrect, shade by probability
        correct = row.correct
        is_ad   = row.true_label == "AD"
        color   = "#2ecc71" if correct else "#e74c3c"
        alpha   = 0.4 + 0.6 * abs(row.ad_probability - 0.5) * 2

        rect = plt.Rectangle(
            (col_idx + 0.05, row_idx + 0.05), 0.9, 0.8,
            facecolor=color, alpha=alpha, edgecolor="white", linewidth=1
        )
        ax.add_patch(rect)
        ax.text(col_idx + 0.5, row_idx + 0.45,
                f"{row.subject_id}\n{'AD' if is_ad else 'HC'}",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Correct"),
        Patch(facecolor="#e74c3c", label="Incorrect"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1, 1.05))
    ax.set_title("Zero-Shot Per-Subject Predictions — ds006036 (Eyes-Open)\n"
                 "Shade intensity = distance from decision threshold",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Override subject list (default: 1-65)")
    parser.add_argument("--model", default=str(MODEL_PATH),
                        help="Path to trained model .joblib")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    subjects  = args.subjects or SUBJECTS
    threshold = args.threshold
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Zero-Shot Evaluation — ds006036 (Eyes-Open)")
    print(f"  Model: {model_path}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Subjects: {len(subjects)}")
    print(f"{'='*60}\n")

    model = joblib.load(model_path)

    # Run inference on all subjects
    rows = []
    for subject_id in subjects:
        print(f"Subject {subject_id:02d}:", end=" ")
        result = run_subject(subject_id, model, data_root=args.data_root)
        if result is None:
            print("skipped")
            continue
        correct_str = "✓" if result["correct"] else "✗"
        print(f"{result['true_label']} → {result['prediction']} "
              f"(p={result['ad_probability']:.3f}) {correct_str}")
        rows.append({k: v for k, v in result.items() if k != "window_probs"})

    df = pd.DataFrame(rows)

    # Subject-level metrics
    all_probs = df["ad_probability"].values
    all_true  = (df["true_label"] == "AD").astype(int).values
    all_pred  = (df["prediction"] == "AD").astype(int).values

    subject_auc  = roc_auc_score(all_true, all_probs)
    accuracy     = accuracy_score(all_true, all_pred)
    f1           = f1_score(all_true, all_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    df["sensitivity"] = sensitivity
    df["specificity"] = specificity

    print(f"\n{'='*60}")
    print(f"  ZERO-SHOT RESULTS (subject-level)")
    print(f"  AUC:         {subject_auc:.3f}")
    print(f"  Accuracy:    {accuracy:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  F1:          {f1:.3f}")
    print(f"  Correct:     {int(df['correct'].sum())}/{len(df)}")
    print(f"{'='*60}\n")

    # Save results CSV
    csv_path = OUTPUT_DIR / "zeroshot_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Figures
    print("\nGenerating figures:")
    plot_roc(df, all_probs, all_true,
             OUTPUT_DIR / "figures" / "1_zeroshot_roc.png")
    plot_probability_distribution(df,
             OUTPUT_DIR / "figures" / "2_probability_distribution.png")
    plot_subject_heatmap(df,
             OUTPUT_DIR / "figures" / "3_subject_predictions.png")

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()