"""
Generate all evaluation figures for the final bandpower RF model.

Usage:
    python scripts/evaluate.py

Reads from:
    outputs/models/bandpower_rf_thresh_v1/model.joblib
    outputs/results/bandpower_rf_thresh_v1/fold_metrics.csv
    outputs/features/bandpower_v1/

Writes figures to:
    outputs/results/bandpower_rf_thresh_v1/figures/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import BandpowerDataset, bandpower_to_numpy
from src.evaluation.importance import (
    compute_importances_matrix, electrode_sweep, load_channel_ranking, CHANNEL_NAMES, BANDS
)
from src.evaluation.plots import plot_electrode_sweep
from src.utils import set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH      = Path("outputs/models/bandpower_rf_thresh_v1/model.joblib")
FOLD_CSV        = Path("outputs/results/bandpower_rf_thresh_v1/fold_metrics.csv")
FEATURES_DIR    = Path("outputs/features/bandpower_v1")
FIGURES_DIR     = Path("outputs/results/bandpower_rf_thresh_v1/figures")
SWEEP_CSV       = Path("outputs/results/bandpower_rf_thresh_v1/electrode_sweep.csv")

CHANNELS        = [12, 14, 9, 16, 10]   # T4, Pz, Cz, T6, C3
SUBJECTS        = list(range(1, 66))
N_SPLITS        = 5
RF_PARAMS       = {"n_estimators": 200, "max_depth": None, "min_samples_split": 2}

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BLUE   = "#4DA8DA"
RED    = "#E05C5C"
GOLD   = "#F5A623"
GRAY   = "#888888"


def load_data():
    """Load features and build aligned (X, y, subject_ids) arrays."""
    X, y = bandpower_to_numpy(
        str(FEATURES_DIR), SUBJECTS, ["AD", "HC"], CHANNELS
    )
    ds = BandpowerDataset(
        str(FEATURES_DIR), subjects=SUBJECTS,
        diagnosis_filter=["AD", "HC"], channels=CHANNELS
    )
    subject_ids = np.zeros(len(ds), dtype=int)
    cumulative = 0
    for s in ds.samples:
        subject_ids[cumulative: cumulative + s["n_windows"]] = s["subject_id"]
        cumulative += s["n_windows"]
    return X, y, subject_ids


# ---------------------------------------------------------------------------
# 1. ROC curve — one curve per fold + mean
# ---------------------------------------------------------------------------
def plot_roc(X, y, subject_ids, save_path):
    print("  Generating ROC curve...")
    set_seed(42)
    gkf = GroupKFold(n_splits=N_SPLITS)

    fig, ax = plt.subplots(figsize=(7, 6))
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 200)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subject_ids)):
        rf = RandomForestClassifier(random_state=42 + fold, n_jobs=-1, **RF_PARAMS)
        rf.fit(X[train_idx], y[train_idx])
        y_prob = rf.predict_proba(X[test_idx])[:, 1]

        fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, alpha=0.3, linewidth=1, color=BLUE,
                label=f"Fold {fold+1} (AUC={fold_auc:.3f})" if fold == 0 else f"Fold {fold+1} (AUC={fold_auc:.3f})")

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color=RED, linewidth=2.5,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    alpha=0.15, color=RED, label="± 1 std")
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1, label="Chance")

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — Bandpower RF (5 Electrodes, Subject-wise CV)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Confusion matrix at optimized threshold (averaged across folds)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(fold_df, save_path):
    print("  Generating confusion matrix...")

    tp = int(fold_df["tp"].mean())
    tn = int(fold_df["tn"].mean())
    fp = int(fold_df["fp"].mean())
    fn = int(fold_df["fn"].mean())

    cm = np.array([[tp, fn],
                   [fp, tn]])

    # Percentages for annotation
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct, annot=False, fmt=".1f", cmap="Blues",
        xticklabels=["Predicted AD", "Predicted HC"],
        yticklabels=["Actual AD", "Actual HC"],
        ax=ax, linewidths=0.5, linecolor="white",
        vmin=0, vmax=100,
    )
    # Manual annotations: count + percentage
    labels = [[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, labels[i][j],
                    ha="center", va="center", fontsize=13,
                    color="white" if cm_pct[i, j] > 60 else "black")

    mean_sens = fold_df["sensitivity"].mean()
    mean_spec = fold_df["specificity"].mean()
    mean_thr  = fold_df["threshold"].mean()

    ax.set_title(
        f"Confusion Matrix (avg. across folds)\n"
        f"Threshold={mean_thr:.3f}  Sensitivity={mean_sens:.1%}  Specificity={mean_spec:.1%}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3. Per-fold AUC box plot
# ---------------------------------------------------------------------------
def plot_per_fold_auc(fold_df, save_path):
    print("  Generating per-fold AUC plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    aucs = fold_df["auc"].values
    folds = fold_df["fold"].values
    mean_auc = aucs.mean()

    bars = ax.bar(folds, aucs, color=BLUE, alpha=0.8,
                  edgecolor="navy", linewidth=1, width=0.5)
    ax.axhline(mean_auc, color=RED, linestyle="--", linewidth=2,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1, label="Chance")

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xlabel("CV Fold")
    ax.set_ylabel("AUC")
    ax.set_title("Per-Fold AUC — Subject-wise 5-Fold CV")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# 4. Feature importance heatmap (channel × band)
# ---------------------------------------------------------------------------
def plot_importance_heatmap(save_path):
    print("  Generating feature importance heatmap...")

    from matplotlib.colors import LinearSegmentedColormap
    rf = joblib.load(MODEL_PATH)
    imp = rf.feature_importances_  # shape: (25,) for 5ch × 5bands

    n_ch   = len(CHANNELS)
    n_bands = len(BANDS)
    imp_matrix = imp.reshape(n_ch, n_bands)

    # Normalize 0-1
    lo, hi = imp_matrix.min(), imp_matrix.max()
    imp_norm = (imp_matrix - lo) / (hi - lo) if hi > lo else np.zeros_like(imp_matrix)

    ch_labels = [CHANNEL_NAMES[i] for i in CHANNELS]
    df_imp = pd.DataFrame(imp_norm, index=ch_labels, columns=BANDS)

    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#00aa00", "#ffff00", "#dd0000"], N=256
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df_imp, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Normalized Gini Importance (0–1)"},
        ax=ax,
    )
    ax.set_title(
        "Bandpower Feature Importance by Electrode and Frequency Band\n"
        "(Random Forest, Final Model Trained on All 65 Subjects)"
    )
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("EEG Electrode")
    ax.set_xticklabels(BANDS, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# 5. Sensitivity / specificity tradeoff curve
# ---------------------------------------------------------------------------
def plot_threshold_tradeoff(X, y, subject_ids, save_path):
    print("  Generating sensitivity/specificity tradeoff curve...")
    set_seed(42)
    gkf = GroupKFold(n_splits=N_SPLITS)

    # Collect all OOF probabilities
    all_probs = np.zeros(len(y))
    all_true  = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subject_ids)):
        rf = RandomForestClassifier(random_state=42 + fold, n_jobs=-1, **RF_PARAMS)
        rf.fit(X[train_idx], y[train_idx])
        all_probs[test_idx] = rf.predict_proba(X[test_idx])[:, 1]
        all_true[test_idx]  = y[test_idx]

    fpr, tpr, thresholds = roc_curve(all_true, all_probs)
    specificity = 1.0 - fpr

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, tpr,         color=RED,  linewidth=2, label="Sensitivity")
    ax.plot(thresholds, specificity, color=BLUE, linewidth=2, label="Specificity")

    # Mark the mean optimized threshold
    mean_thr = pd.read_csv(FOLD_CSV)["threshold"].mean()
    ax.axvline(mean_thr, color=GOLD, linestyle="--", linewidth=2,
               label=f"Selected threshold ({mean_thr:.3f})")

    # Annotate operating point
    idx = np.argmin(np.abs(thresholds - mean_thr))
    ax.annotate(
        f"Sens={tpr[idx]:.2f}\nSpec={specificity[idx]:.2f}",
        xy=(mean_thr, tpr[idx]),
        xytext=(mean_thr + 0.08, tpr[idx] - 0.12),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Sensitivity / Specificity Tradeoff\n(Out-of-Fold Probabilities, All 5 Folds)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# 6. Electrode sweep
# ---------------------------------------------------------------------------
def plot_sweep(save_path):
    print("  Generating electrode sweep...")

    if not SWEEP_CSV.exists():
        print("    Sweep CSV not found — running electrode sweep (this takes a few minutes)...")
        channel_ranking = load_channel_ranking(str(MODEL_PATH), trained_on_channels=CHANNELS)
        df = electrode_sweep(
            features_dir=str(FEATURES_DIR),
            subjects=SUBJECTS,
            channel_ranking=channel_ranking,
            top_k_list=[1, 3, 5, 8, 12, 19],
            rf_params=RF_PARAMS,
            output_dir=str(SWEEP_CSV.parent),
        )
    else:
        print("    Loading existing sweep CSV...")
        df = pd.read_csv(SWEEP_CSV)

    fig, ax = plt.subplots(figsize=(9, 5))

    baseline = float(df.loc[df["n_channels"] == 19, "auc_mean"].values[0])
    best_k   = int(df.loc[df["auc_mean"].idxmax(), "n_channels"])
    best_auc = float(df["auc_mean"].max())

    ax.errorbar(
        df["n_channels"], df["auc_mean"], df["auc_std"],
        fmt="o-", capsize=5, linewidth=2, color=BLUE,
        markersize=7, label="CV AUC ± std",
    )
    ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1.5,
               label=f"Full 19-electrode baseline ({baseline:.3f})")
    ax.scatter([best_k], [best_auc], color=RED, s=120, zorder=5,
               label=f"Best: k={best_k} ({best_auc:.3f})")

    ax.set_xlabel("Number of Electrodes")
    ax.set_ylabel("Cross-Validation AUC")
    ax.set_title("Electrode Reduction Sweep\n(Fixed GroupKFold Splits, Bandpower + RF)")
    ax.set_xticks(df["n_channels"].tolist())
    ax.legend()
    ax.set_ylim(bottom=max(0.4, float(df["auc_mean"].min()) - 0.05))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading fold metrics...")
    fold_df = pd.read_csv(FOLD_CSV)
    print(f"  {len(fold_df)} folds loaded\n")

    print("Loading features for OOF evaluation...")
    X, y, subject_ids = load_data()
    print()

    print("Generating figures:")
    plot_roc(X, y, subject_ids,
             FIGURES_DIR / "1_roc_curve.png")
    plot_confusion_matrix(fold_df,
             FIGURES_DIR / "2_confusion_matrix.png")
    plot_per_fold_auc(fold_df,
             FIGURES_DIR / "3_per_fold_auc.png")
    plot_importance_heatmap(
             FIGURES_DIR / "4_feature_importance.png")
    plot_threshold_tradeoff(X, y, subject_ids,
             FIGURES_DIR / "5_threshold_tradeoff.png")
    plot_sweep(
             FIGURES_DIR / "6_electrode_sweep.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("  1_roc_curve.png")
    print("  2_confusion_matrix.png")
    print("  3_per_fold_auc.png")
    print("  4_feature_importance.png")
    print("  5_threshold_tradeoff.png")
    print("  6_electrode_sweep.png")


if __name__ == "__main__":
    main()
