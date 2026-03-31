"""
Generate all evaluation figures for a bandpower RF model run.

Usage:
    python scripts/evaluate.py configs/evaluation/bandpower_rf_thresh_v1.yaml
    python scripts/evaluate.py configs/evaluation/bandpower_rf_thresh_ica_v1.yaml

All paths and parameters are read from the evaluation config YAML.
To evaluate a new run, just create a new YAML in configs/evaluation/.
"""

import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import BandpowerDataset, bandpower_to_numpy
from src.evaluation.importance import (
    electrode_sweep, load_channel_ranking, CHANNEL_NAMES, BANDS
)
from src.utils import set_seed

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
BLUE = "#4DA8DA"
RED  = "#E05C5C"
GOLD = "#F5A623"
GRAY = "#888888"


def load_data(cfg):
    X, y = bandpower_to_numpy(
        cfg["features_dir"], cfg["subjects"], ["AD", "HC"], cfg["channels"]
    )
    ds = BandpowerDataset(
        cfg["features_dir"], subjects=cfg["subjects"],
        diagnosis_filter=["AD", "HC"], channels=cfg["channels"],
    )
    subject_ids = np.zeros(len(ds), dtype=int)
    cumulative = 0
    for s in ds.samples:
        subject_ids[cumulative: cumulative + s["n_windows"]] = s["subject_id"]
        cumulative += s["n_windows"]
    return X, y, subject_ids


def _make_rf(cfg, fold):
    p = cfg["rf_params"]
    return RandomForestClassifier(
        random_state=42 + fold, n_jobs=-1,
        n_estimators=p["n_estimators"],
        max_depth=p.get("max_depth"),
        min_samples_split=p.get("min_samples_split", 2),
    )


def plot_roc(X, y, subject_ids, cfg, save_path):
    print("  Generating ROC curve...")
    set_seed(42)
    fig, ax = plt.subplots(figsize=(7, 6))
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 200)

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=cfg["n_splits"]).split(X, y, subject_ids)
    ):
        rf = _make_rf(cfg, fold)
        rf.fit(X[train_idx], y[train_idx])
        y_prob = rf.predict_proba(X[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, alpha=0.3, linewidth=1, color=BLUE,
                label=f"Fold {fold+1} (AUC={fold_auc:.3f})")

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color=RED, linewidth=2.5,
            label=f"Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    alpha=0.15, color=RED, label="+/- 1 std")
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve -- {cfg['name']}\n({len(cfg['channels'])} Electrodes, Subject-wise CV)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_confusion_matrix(fold_df, save_path):
    print("  Generating confusion matrix...")
    tp = int(fold_df["tp"].mean())
    tn = int(fold_df["tn"].mean())
    fp = int(fold_df["fp"].mean())
    fn = int(fold_df["fn"].mean())
    cm = np.array([[tp, fn], [fp, tn]])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=False, cmap="Blues",
                xticklabels=["Predicted AD", "Predicted HC"],
                yticklabels=["Actual AD", "Actual HC"],
                ax=ax, linewidths=0.5, linecolor="white", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm_pct[i, j] > 60 else "black")
    ax.set_title(
        f"Confusion Matrix (avg. across folds)\n"
        f"Threshold={fold_df['threshold'].mean():.3f}  "
        f"Sensitivity={fold_df['sensitivity'].mean():.1%}  "
        f"Specificity={fold_df['specificity'].mean():.1%}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_per_fold_auc(fold_df, save_path):
    print("  Generating per-fold AUC plot...")
    aucs  = fold_df["auc"].values
    folds = fold_df["fold"].values
    mean_auc = aucs.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(folds, aucs, color=BLUE, alpha=0.8,
                  edgecolor="navy", linewidth=1, width=0.5)
    ax.axhline(mean_auc, color=RED, linestyle="--", linewidth=2,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1, label="Chance")
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("AUC")
    ax.set_title("Per-Fold AUC -- Subject-wise 5-Fold CV")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_importance_heatmap(cfg, save_path):
    print("  Generating feature importance heatmap...")
    from matplotlib.colors import LinearSegmentedColormap

    rf       = joblib.load(cfg["model_path"])
    channels = cfg["channels"]
    n_ch     = len(channels)
    n_bands  = len(BANDS)

    imp        = rf.feature_importances_
    imp_matrix = imp.reshape(n_ch, n_bands)
    lo, hi     = imp_matrix.min(), imp_matrix.max()
    imp_norm   = (imp_matrix - lo) / (hi - lo) if hi > lo else np.zeros_like(imp_matrix)

    ch_labels = [CHANNEL_NAMES[i] for i in channels]
    df_imp    = pd.DataFrame(imp_norm, index=ch_labels, columns=BANDS)
    cmap      = LinearSegmentedColormap.from_list(
        "rg", ["#00aa00", "#ffff00", "#dd0000"], N=256
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_imp, cmap=cmap, vmin=0, vmax=1,
                annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Normalized Gini Importance (0-1)"}, ax=ax)
    ax.set_title(
        f"Bandpower Feature Importance -- {cfg['name']}\n"
        "(Random Forest, Final Model Trained on All 65 Subjects)"
    )
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("EEG Electrode")
    ax.set_xticklabels(BANDS, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_threshold_tradeoff(X, y, subject_ids, cfg, save_path):
    print("  Generating sensitivity/specificity tradeoff curve...")
    set_seed(42)
    all_probs = np.zeros(len(y))
    all_true  = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=cfg["n_splits"]).split(X, y, subject_ids)
    ):
        rf = _make_rf(cfg, fold)
        rf.fit(X[train_idx], y[train_idx])
        all_probs[test_idx] = rf.predict_proba(X[test_idx])[:, 1]
        all_true[test_idx]  = y[test_idx]

    fpr, tpr, thresholds = roc_curve(all_true, all_probs)
    specificity = 1.0 - fpr
    mean_thr    = pd.read_csv(cfg["fold_csv"])["threshold"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, tpr,         color=RED,  linewidth=2, label="Sensitivity")
    ax.plot(thresholds, specificity, color=BLUE, linewidth=2, label="Specificity")
    ax.axvline(mean_thr, color=GOLD, linestyle="--", linewidth=2,
               label=f"Selected threshold ({mean_thr:.3f})")
    idx = np.argmin(np.abs(thresholds - mean_thr))
    ax.annotate(
        f"Sens={tpr[idx]:.2f}\nSpec={specificity[idx]:.2f}",
        xy=(mean_thr, tpr[idx]),
        xytext=(mean_thr + 0.08, tpr[idx] - 0.12),
        arrowprops=dict(arrowstyle="->", color="black"), fontsize=10,
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


def plot_sweep(cfg, save_path):
    print("  Generating electrode sweep...")
    sweep_csv = Path(cfg["sweep_csv"])

    if not sweep_csv.exists():
        print("    Sweep CSV not found -- running electrode sweep (takes a few minutes)...")
        channel_ranking = load_channel_ranking(
            cfg["model_path"], trained_on_channels=cfg["channels"]
        )
        p = cfg["rf_params"]
        df = electrode_sweep(
            features_dir=cfg["features_dir"],
            subjects=cfg["subjects"],
            channel_ranking=channel_ranking,
            top_k_list=[1, 3, 5, 8, 12, 19],
            rf_params={
                "n_estimators": p["n_estimators"],
                "max_depth": p.get("max_depth"),
                "min_samples_split": p.get("min_samples_split", 2),
            },
            output_dir=str(sweep_csv.parent),
        )
    else:
        print("    Loading existing sweep CSV...")
        df = pd.read_csv(sweep_csv)

    baseline = float(df.loc[df["n_channels"] == 19, "auc_mean"].values[0])
    best_k   = int(df.loc[df["auc_mean"].idxmax(), "n_channels"])
    best_auc = float(df["auc_mean"].max())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(df["n_channels"], df["auc_mean"], df["auc_std"],
                fmt="o-", capsize=5, linewidth=2, color=BLUE,
                markersize=7, label="CV AUC +/- std")
    ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1.5,
               label=f"Full 19-electrode baseline ({baseline:.3f})")
    ax.scatter([best_k], [best_auc], color=RED, s=120, zorder=5,
               label=f"Best: k={best_k} ({best_auc:.3f})")
    ax.set_xlabel("Number of Electrodes")
    ax.set_ylabel("Cross-Validation AUC")
    ax.set_title(f"Electrode Reduction Sweep -- {cfg['name']}\n"
                 "(Fixed GroupKFold Splits, Bandpower + RF)")
    ax.set_xticks(df["n_channels"].tolist())
    ax.legend()
    ax.set_ylim(bottom=max(0.4, float(df["auc_mean"].min()) - 0.05))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to evaluation config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Evaluating: {cfg['name']}")
    if cfg.get("description"):
        print(f"  {cfg['description']}")
    print(f"{'='*55}\n")

    print("Loading fold metrics...")
    fold_df = pd.read_csv(cfg["fold_csv"])
    print(f"  {len(fold_df)} folds loaded\n")

    print("Loading features for OOF evaluation...")
    X, y, subject_ids = load_data(cfg)
    print()

    print("Generating figures:")
    plot_roc(X, y, subject_ids, cfg,          figures_dir / "1_roc_curve.png")
    plot_confusion_matrix(fold_df,            figures_dir / "2_confusion_matrix.png")
    plot_per_fold_auc(fold_df,                figures_dir / "3_per_fold_auc.png")
    plot_importance_heatmap(cfg,              figures_dir / "4_feature_importance.png")
    plot_threshold_tradeoff(X, y, subject_ids, cfg, figures_dir / "5_threshold_tradeoff.png")
    plot_sweep(cfg,                           figures_dir / "6_electrode_sweep.png")

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()