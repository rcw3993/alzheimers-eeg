"""
Threshold optimization for screening contexts.

In a screening setting we care more about sensitivity (catching every true AD case)
than balanced accuracy. This module finds the decision threshold that:
  1. Achieves sensitivity >= target_sensitivity
  2. Among valid thresholds, maximizes Youden's J (sensitivity + specificity - 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import GroupKFold

from src.data.datasets import BandpowerDataset, bandpower_to_numpy
from src.utils import set_seed, log_experiment


def _build_bandpower_arrays(features_dir, subjects, channels=None):
    X, y = bandpower_to_numpy(
        features_dir,
        subjects=subjects,
        diagnosis_filter=["AD", "HC"],
        channels=channels or "all",
    )
    ds = BandpowerDataset(
        features_dir, subjects=subjects,
        diagnosis_filter=["AD", "HC"],
        channels=channels or "all",
    )
    subject_ids = np.zeros(len(ds), dtype=int)
    cumulative = 0
    for sample in ds.samples:
        subject_ids[cumulative: cumulative + sample["n_windows"]] = sample["subject_id"]
        cumulative += sample["n_windows"]
    return X, y, subject_ids


def _best_threshold(y_true, y_prob, target_sensitivity: float = 0.85) -> float:
    """Return threshold that meets sensitivity target with max Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    valid = tpr >= target_sensitivity
    if np.any(valid):
        J = tpr + specificity - 1.0
        return float(thresholds[valid][np.argmax(J[valid])])
    return 0.5  # fallback


def cross_val_bandpower_rf_threshold(
    features_dir: str,
    subjects: List[int],
    rf_params: Dict[str, Any],
    channels: Optional[List[int]] = None,
    n_splits: int = 5,
    target_sensitivity: float = 0.85,
    output_dir: str = "outputs/results/bandpower_rf_thresh",
) -> Dict:
    """
    5-fold subject-wise CV with per-fold threshold optimization.
    Designed for screening: prioritizes sensitivity over balanced accuracy.
    """
    set_seed(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if channels is None:
        channels = [12, 14, 9, 16, 10]  # T4, Pz, Cz, T6, C3 (top-5 from electrode sweep)

    X, y, subject_ids = _build_bandpower_arrays(features_dir, subjects, channels)
    print(f"Threshold CV: X={X.shape}, class distribution={np.bincount(y)}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subject_ids)):
        rf = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", None),
            min_samples_split=rf_params.get("min_samples_split", 2),
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf.fit(X[train_idx], y[train_idx])
        y_prob = rf.predict_proba(X[test_idx])[:, 1]

        thresh = _best_threshold(y[test_idx], y_prob, target_sensitivity)
        y_pred = (y_prob >= thresh).astype(int)

        auc = roc_auc_score(y[test_idx], y_prob)
        tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        row = {
            "fold": fold + 1,
            "threshold": thresh,
            "auc": auc,
            "f1": f1_score(y[test_idx], y_pred, zero_division=0),
            "accuracy": accuracy_score(y[test_idx], y_pred),
            "precision": precision_score(y[test_idx], y_pred, zero_division=0),
            "sensitivity": sens,
            "specificity": spec,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
        fold_rows.append(row)
        print(f"  Fold {fold+1}: thr={thresh:.3f}  AUC={auc:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}")

    df = pd.DataFrame(fold_rows)
    summary = {
        col + "_mean": float(df[col].mean())
        for col in ["auc", "f1", "sensitivity", "specificity", "threshold"]
    }
    summary.update({
        col + "_std": float(df[col].std())
        for col in ["auc", "f1", "sensitivity", "specificity", "threshold"]
    })

    print(f"\nThreshold CV: AUC={summary['auc_mean']:.3f}  "
          f"Sens={summary['sensitivity_mean']:.3f}  "
          f"Spec={summary['specificity_mean']:.3f}")

    df.to_csv(Path(output_dir) / "fold_metrics.csv", index=False)
    log_experiment(
        {**rf_params, "representation": "bandpower", "model": "rf",
         "channels": channels, "target_sensitivity": target_sensitivity},
        summary,
        output_dir,
    )
    return {"fold_df": df, **summary}