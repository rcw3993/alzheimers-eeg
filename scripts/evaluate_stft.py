"""
Electrode sweep for STFT + 2D CNN, using the bandpower RF channel ranking
as a proxy for electrode importance.

Usage:
    python scripts/evaluate_stft.py

Reads from:
    outputs/features/stft_v1/
    outputs/models/bandpower_rf_thresh_v1/model.channel_imp.npy  (for ranking)

Writes to:
    outputs/results/stft_cnn_sweep/
        electrode_sweep.csv
        figures/stft_electrode_sweep.png
        figures/stft_vs_bp_comparison.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import STFTDataset
from src.evaluation.importance import load_channel_ranking, CHANNEL_NAMES
from src.models.cnn2d import Simple2DCNN
from src.utils import set_seed

# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------
STFT_FEATURES_DIR = Path("outputs/features/stft_v1")
BP_MODEL_PATH     = Path("outputs/models/bandpower_rf_thresh_v1/model.joblib")
BP_CHANNELS       = [12, 14, 9, 16, 10]   # what the BP model was trained on
OUTPUT_DIR        = Path("outputs/results/stft_cnn_sweep")
SWEEP_CSV         = OUTPUT_DIR / "electrode_sweep.csv"
BP_SWEEP_CSV      = Path("outputs/results/bandpower_rf_thresh_v1/electrode_sweep.csv")

SUBJECTS          = list(range(1, 66))
TOP_K_LIST        = [1, 3, 5, 8, 12, 19]
N_SPLITS          = 5
N_PER_SUBJECT     = 32
EPOCHS            = 15
LR                = 1e-3
BATCH_SIZE        = 64

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
# Data loading
# ---------------------------------------------------------------------------
def smart_sample_stft(channels):
    """Load first N_PER_SUBJECT windows per subject for given channel subset."""
    ds = STFTDataset(
        str(STFT_FEATURES_DIR),
        subjects=SUBJECTS,
        diagnosis_filter=["AD", "HC"],
        channels=channels,
    )
    all_X, all_y, all_ids = [], [], []
    for sample in ds.samples:
        tensor = torch.load(sample["path"], weights_only=True)
        n = min(N_PER_SUBJECT, tensor.shape[0])
        # Select the requested channels
        ch_tensor = torch.tensor(channels)
        all_X.append(tensor[:n][:, ch_tensor, :, :])
        label = 0 if sample["diagnosis"] == "AD" else 1
        all_y.extend([label] * n)
        all_ids.extend([sample["subject_id"]] * n)

    return torch.cat(all_X), torch.tensor(all_y), np.array(all_ids)


# ---------------------------------------------------------------------------
# Training + evaluation for one fold
# ---------------------------------------------------------------------------
def train_eval_fold(X_train, y_train, X_test, y_test, device, fold):
    model = Simple2DCNN(n_channels=X_train.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        out = model(X_test.to(device))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

    return float(roc_auc_score(y_test.numpy(), probs))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(channel_ranking, device):
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate fixed splits from full 19-channel data once
    print("Loading full 19-channel STFT data for fixed split generation...")
    X_full, y_full, subject_ids = smart_sample_stft(list(range(19)))
    cv_splits = list(
        GroupKFold(n_splits=N_SPLITS).split(X_full, y_full, subject_ids)
    )
    print(f"Fixed {len(cv_splits)} splits, {X_full.shape[0]} total windows\n")

    rows = []
    for k in TOP_K_LIST:
        channels = channel_ranking[:k].tolist()
        print(f"k={k:2d}: channels={[CHANNEL_NAMES[c] for c in channels]}")

        X_k, y_k, _ = smart_sample_stft(channels)
        print(f"       X_k shape: {tuple(X_k.shape)}")

        aucs = []
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            set_seed(42 + fold)
            auc = train_eval_fold(
                X_k[train_idx], y_k[train_idx],
                X_k[test_idx],  y_k[test_idx],
                device, fold,
            )
            aucs.append(auc)
            print(f"         Fold {fold+1}: AUC={auc:.3f}")

        rows.append({
            "n_channels": k,
            "channels":   channels,
            "auc_mean":   float(np.mean(aucs)),
            "auc_std":    float(np.std(aucs)),
        })
        print(f"       Mean: {rows[-1]['auc_mean']:.3f} +/- {rows[-1]['auc_std']:.3f}\n")

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    print(f"Sweep saved to {SWEEP_CSV}")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_stft_sweep(df, save_path):
    baseline = float(df.loc[df["n_channels"] == 19, "auc_mean"].values[0])
    best_k   = int(df.loc[df["auc_mean"].idxmax(), "n_channels"])
    best_auc = float(df["auc_mean"].max())

    fig, ax = plt.subplots(figsize=(9, 5))
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
    ax.set_title("Electrode Reduction Sweep — STFT + 2D CNN\n"
                 "(Channel ranking from Bandpower RF Gini importance)")
    ax.set_xticks(df["n_channels"].tolist())
    ax.legend()
    ax.set_ylim(bottom=max(0.4, float(df["auc_mean"].min()) - 0.05))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(df_stft, save_path):
    """Overlay bandpower RF sweep and STFT CNN sweep on the same axes."""
    if not BP_SWEEP_CSV.exists():
        print(f"Bandpower sweep CSV not found at {BP_SWEEP_CSV}, skipping comparison plot.")
        return

    df_bp = pd.read_csv(BP_SWEEP_CSV)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_bp["n_channels"], df_bp["auc_mean"], df_bp["auc_std"],
        fmt="s-", capsize=4, linewidth=2, color=GOLD,
        markersize=7, label="Bandpower + RF",
    )
    ax.errorbar(
        df_stft["n_channels"], df_stft["auc_mean"], df_stft["auc_std"],
        fmt="o-", capsize=4, linewidth=2, color=BLUE,
        markersize=7, label="STFT + 2D CNN",
    )
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1, label="Chance")
    ax.set_xlabel("Number of Electrodes")
    ax.set_ylabel("Cross-Validation AUC")
    ax.set_title("Electrode Reduction: Bandpower RF vs STFT CNN\n"
                 "(Same channel ranking, same fixed GroupKFold splits)")
    ax.set_xticks(sorted(set(
        df_bp["n_channels"].tolist() + df_stft["n_channels"].tolist()
    )))
    ax.legend()
    ax.set_ylim(bottom=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load channel ranking from the bandpower RF model
    print("Loading channel ranking from bandpower RF model...")
    channel_ranking = load_channel_ranking(
        str(BP_MODEL_PATH),
        trained_on_channels=BP_CHANNELS,
    )
    print(f"Ranking (top 10): {[CHANNEL_NAMES[i] for i in channel_ranking[:10]]}\n")

    # Run or load sweep
    if SWEEP_CSV.exists():
        print(f"Found existing sweep CSV at {SWEEP_CSV}, loading...")
        df = pd.read_csv(SWEEP_CSV)
    else:
        df = run_sweep(channel_ranking, device)

    print("\nSweep results:")
    print(df[["n_channels", "auc_mean", "auc_std"]].to_string(index=False))

    print("\nGenerating figures...")
    plot_stft_sweep(df, figures_dir / "stft_electrode_sweep.png")
    plot_comparison(df, figures_dir / "stft_vs_bp_comparison.png")

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()