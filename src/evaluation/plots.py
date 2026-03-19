"""
All plotting functions for the EEG-Alzheimer's pipeline.

Every function accepts explicit data arguments and an output path.
No hardcoded file paths or result directories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Optional, Tuple


CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4", "T5", "P3",
    "Pz",  "P4",  "T6", "O1", "O2",
]


def plot_electrode_sweep(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    baseline_k: int = 19,
):
    """Line plot of AUC vs number of electrodes with error bars."""
    baseline = df.loc[df["n_channels"] == baseline_k, "auc_mean"].values[0]

    plt.figure(figsize=(10, 6))
    plt.errorbar(df["n_channels"], df["auc_mean"], df["auc_std"],
                 fmt="o-", capsize=5, linewidth=2)
    plt.axhline(baseline, color="r", linestyle="--",
                label=f"Full {baseline_k}ch ({baseline:.3f})")
    plt.xlabel("Number of Electrodes")
    plt.ylabel("Cross-Validation AUC")
    plt.title("Electrode Reduction: Bandpower + RF (Fixed Splits)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_feature_importance_bar(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 10,
    save_path: Optional[str] = None,
    color: str = "#4DA8DA",
):
    """Horizontal bar chart of top-N RF feature importances."""
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    plt.figure(figsize=(10, max(4, top_n * 0.6)))
    plt.barh(range(top_n), top_imp, color=color, alpha=0.8, edgecolor="navy", linewidth=1)
    plt.yticks(range(top_n), top_names)
    plt.xlabel("Feature Importance (Gini)", fontsize=12, fontweight="bold")
    plt.title(f"Top {top_n} Bandpower Features\n(Random Forest, Subject-wise CV)",
              fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_importance_heatmap(
    df_imp: pd.DataFrame,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Heatmap of normalized [channel × band] importances.
    Red = high importance, green = low importance.
    """
    channel_names = channel_names or CHANNEL_NAMES[: len(df_imp)]
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#00ff00", "#ffff00", "#ff0000"], N=256
    )

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        df_imp.values,
        xticklabels=df_imp.columns,
        yticklabels=channel_names,
        cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Normalized Gini Importance (0–1)"},
        linewidths=0.5, linecolor="white",
    )
    plt.title(
        "Bandpower Feature Importance by Electrode and Frequency Band\n"
        "(Random Forest Gini Importance, Normalized 0–1)",
        fontsize=14, fontweight="bold", pad=16,
    )
    plt.xlabel("Frequency Band", fontsize=12, fontweight="bold")
    plt.ylabel("EEG Electrode", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_or_show(save_path, dpi=dpi)


def plot_representation_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Side-by-side bar charts: AUC comparison + feature count (log scale).
    results_df must have columns: Representation, AUC, Features.
    """
    colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#A9A9A9"][: len(results_df)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars = ax1.bar(results_df["Representation"], results_df["AUC"],
                   color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
    best_auc = results_df["AUC"].max()
    ax1.axhline(best_auc, color="red", linestyle="--", linewidth=2,
                label=f"Best ({best_auc:.3f})")
    for bar, auc in zip(bars, results_df["AUC"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{auc:.3f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Cross-Validation AUC", fontsize=12, fontweight="bold")
    ax1.set_title("Representation Comparison\n(Subject-wise 5-fold CV)",
                  fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_ylim(0.45, min(1.0, best_auc + 0.1))
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(results_df)), results_df["Features"],
            color=colors, alpha=0.8, edgecolor="black")
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df["Representation"], rotation=0)
    ax2.set_ylabel("Number of Features (log scale)", fontsize=12, fontweight="bold")
    ax2.set_title("Computational Cost", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_bandpower_subject(
    tensor,  # [n_windows, n_channels, n_bands] tensor or ndarray
    subject_id: int,
    diagnosis: str,
    n_windows: int = 5,
    save_path: Optional[str] = None,
):
    """Heatmap of first n_windows × all features for a single subject."""
    import torch
    if isinstance(tensor, torch.Tensor):
        data = tensor[:n_windows].numpy()
    else:
        data = tensor[:n_windows]
    data_2d = data.reshape(n_windows, -1)

    plt.figure(figsize=(12, 6))
    plt.imshow(data_2d, aspect="auto", cmap="viridis")
    plt.title(f"Bandpower: Sub {subject_id} ({diagnosis})")
    plt.xlabel("Channel × Band  (δ θ α β γ per channel)")
    plt.ylabel(f"Window (0–{n_windows-1})")
    plt.colorbar(label="Power (μV²/Hz)")
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(save_path: Optional[str], dpi: int = 150):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()