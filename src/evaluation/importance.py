"""
Feature importance and electrode reduction analysis.

All functions take explicit paths/parameters — no hardcoded values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data.datasets import BandpowerDataset, bandpower_to_numpy
from src.utils import set_seed


CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4", "T5", "P3",
    "Pz",  "P4",  "T6", "O1", "O2",
]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def load_channel_ranking(model_path: str) -> np.ndarray:
    """
    Load a trained RF model and return channels sorted by importance (highest first).
    Works with both bare joblib models and RandomForestWrapper .npy sidecar files.
    """
    model_path = Path(model_path)
    npy_path = model_path.with_suffix(".channel_imp.npy")

    if npy_path.exists():
        importances = np.load(npy_path)
    else:
        rf = joblib.load(model_path)
        importances = rf.feature_importances_.reshape(19, -1).mean(axis=1)

    return np.argsort(importances)[::-1]


def compute_importances_matrix(
    model_path: str,
    n_channels: int = 19,
    bands: List[str] = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Reshape RF feature importances into a [channel × band] DataFrame,
    normalized to [0, 1].
    """
    bands = bands or BANDS
    rf = joblib.load(model_path)
    imp = rf.feature_importances_.reshape(n_channels, len(bands))

    lo, hi = imp.min(), imp.max()
    imp_norm = (imp - lo) / (hi - lo) if hi > lo else np.zeros_like(imp)

    df = pd.DataFrame(
        imp_norm,
        index=[f"Ch{i:02d}" for i in range(n_channels)],
        columns=bands,
    )
    df.index = CHANNEL_NAMES[:n_channels]

    if save_csv:
        out = Path(model_path).parent / "bandpower_importances_normalized.csv"
        df.to_csv(out)
        print(f"Saved importances to {out}")

    return df


def electrode_sweep(
    features_dir: str,
    subjects: List[int],
    channel_ranking: np.ndarray,
    top_k_list: List[int] = None,
    rf_params: Dict = None,
    output_dir: str = "outputs/results/electrode_sweep",
) -> pd.DataFrame:
    """
    Evaluate bandpower + RF performance as a function of number of electrodes.

    Uses FIXED GroupKFold splits (generated from full 19-channel data) so results
    across different k values are directly comparable.
    """
    set_seed(42)
    top_k_list = top_k_list or [1, 3, 5, 8, 12, 19]
    rf_params = rf_params or {"n_estimators": 200}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate fixed splits once using full-channel data
    X_full, y_full = bandpower_to_numpy(features_dir, subjects, ["AD", "HC"])
    ds_full = BandpowerDataset(features_dir, subjects=subjects, diagnosis_filter=["AD", "HC"])
    subject_ids = np.zeros(len(ds_full), dtype=int)
    cumulative = 0
    for sample in ds_full.samples:
        subject_ids[cumulative: cumulative + sample["n_windows"]] = sample["subject_id"]
        cumulative += sample["n_windows"]

    cv_splits = list(GroupKFold(n_splits=5).split(X_full, y_full, subject_ids))
    print(f"Fixed {len(cv_splits)} splits from {len(subjects)} subjects")

    rows = []
    for k in top_k_list:
        channels = channel_ranking[:k].tolist()
        X_k, _ = bandpower_to_numpy(features_dir, subjects, ["AD", "HC"], channels)

        aucs = []
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            rf = RandomForestClassifier(
                random_state=42 + fold, n_jobs=-1, **rf_params
            )
            rf.fit(X_k[train_idx], y_full[train_idx])
            aucs.append(roc_auc_score(
                y_full[test_idx],
                rf.predict_proba(X_k[test_idx])[:, 1],
            ))

        rows.append({
            "n_channels": k,
            "channels": channels,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
        })
        print(f"  k={k:2d}: {rows[-1]['auc_mean']:.3f} ± {rows[-1]['auc_std']:.3f}")

    df = pd.DataFrame(rows)
    baseline = df.loc[df["n_channels"] == 19, "auc_mean"].values
    if len(baseline):
        df["performance_ratio"] = (df["auc_mean"] / baseline[0]).round(4)

    df.to_csv(Path(output_dir) / "electrode_sweep.csv", index=False)
    print(f"\nElectrode sweep saved to {output_dir}")
    return df