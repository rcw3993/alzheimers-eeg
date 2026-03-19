"""
Subject-wise cross-validation for all representation/model combinations.

All functions follow the same pattern:
  1. Load data with smart sampling (32 windows/subject) where needed
  2. GroupKFold so no subject appears in both train and test
  3. Return a results dict with auc_mean / auc_std
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from src.data.datasets import BandpowerDataset, STFTDataset, ConnectivityDataset, bandpower_to_numpy
from src.models.rf import RandomForestWrapper
from src.models.cnn1d import Simple1DCNN
from src.models.cnn2d import Simple2DCNN
from src.models.mlp import ConnectivityMLP
from src.utils import set_seed, compute_metrics, log_experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_bandpower_arrays(features_dir: str, subjects: List[int], channels=None):
    """Return (X, y, subject_ids) aligned per window for GroupKFold."""
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
        subject_ids[cumulative : cumulative + sample["n_windows"]] = sample["subject_id"]
        cumulative += sample["n_windows"]
    return X, y, subject_ids


def _smart_sample(dataset, n_per_subject: int = 32):
    """Load first n_per_subject windows per subject, return (X_tensor, y_list, subject_ids)."""
    all_X, all_y, all_ids = [], [], []
    for sample in dataset.samples:
        tensor = torch.load(sample["path"], weights_only=True)
        n = min(n_per_subject, tensor.shape[0])
        all_X.append(tensor[:n])
        label = 0 if sample["diagnosis"] == "AD" else 1
        all_y.extend([label] * n)
        all_ids.extend([sample["subject_id"]] * n)
    return torch.cat(all_X), torch.tensor(all_y), np.array(all_ids)


# ---------------------------------------------------------------------------
# Bandpower + Random Forest
# ---------------------------------------------------------------------------

def cross_val_bandpower_rf(
    features_dir: str,
    subjects: List[int],
    rf_params: Dict[str, Any],
    channels: Optional[List[int]] = None,
    n_splits: int = 5,
    output_dir: str = "outputs/results/bandpower_rf",
) -> Dict:
    set_seed(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X, y, subject_ids = _build_bandpower_arrays(features_dir, subjects, channels)
    print(f"Bandpower RF: X={X.shape}, class distribution={np.bincount(y)}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

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
        y_pred = rf.predict(X[test_idx])
        metrics = compute_metrics(y[test_idx], y_prob, y_pred)
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}")

    return _summarize_and_log(fold_metrics, {**rf_params, "representation": "bandpower", "model": "rf"}, output_dir)


def train_final_bandpower_rf(
    features_dir: str,
    subjects: List[int],
    rf_params: Dict[str, Any],
    channels: Optional[List[int]] = None,
    output_dir: str = "outputs/models/bandpower_rf",
) -> RandomForestWrapper:
    """Retrain on full dataset after CV; save model + channel importances."""
    set_seed(42)
    X, y, _ = _build_bandpower_arrays(features_dir, subjects, channels)

    rf = RandomForestWrapper(**rf_params)
    rf.fit(X, y)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rf.save(Path(output_dir) / "model.joblib")
    print(f"Final model saved to {output_dir}")
    print(f"Top 5 channels: {np.argsort(rf.channel_importances)[-5:][::-1]}")
    return rf


# ---------------------------------------------------------------------------
# STFT + 2D CNN
# ---------------------------------------------------------------------------

def cross_val_stft_cnn(
    features_dir: str,
    subjects: List[int],
    channels: Optional[List[int]] = None,
    n_per_subject: int = 32,
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    output_dir: str = "outputs/results/stft_cnn",
) -> Dict:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds = STFTDataset(features_dir, subjects=subjects, diagnosis_filter=["AD", "HC"], channels=channels)
    X_all, y_all, subject_ids = _smart_sample(ds, n_per_subject)
    print(f"STFT CNN: X={X_all.shape}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        train_loader = DataLoader(
            TensorDataset(X_all[train_idx], y_all[train_idx]),
            batch_size=batch_size, shuffle=True,
        )
        model = Simple2DCNN(n_channels=X_all.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

        metrics = _eval_torch(model, X_all[test_idx], y_all[test_idx], device, batch_size)
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}")

    return _summarize_and_log(fold_metrics, {"representation": "stft", "model": "cnn2d"}, output_dir)


# ---------------------------------------------------------------------------
# Connectivity (RF baseline + MLP)
# ---------------------------------------------------------------------------

def cross_val_connectivity(
    features_dir: str,
    subjects: List[int],
    n_per_subject: int = 32,
    n_splits: int = 5,
    mlp_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    rf_n_estimators: int = 200,
    output_dir: str = "outputs/results/connectivity",
) -> Dict:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds = ConnectivityDataset(features_dir, subjects=subjects, diagnosis_filter=["AD", "HC"])
    X_all, y_all, subject_ids = _smart_sample(ds, n_per_subject)
    X_np = X_all.numpy()
    y_np = y_all.numpy()

    gkf = GroupKFold(n_splits=n_splits)
    rf_aucs, mlp_metrics = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        # RF baseline
        rf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42 + fold, n_jobs=-1)
        rf.fit(X_np[train_idx], y_np[train_idx])
        rf_aucs.append(roc_auc_score(y_np[test_idx], rf.predict_proba(X_np[test_idx])[:, 1]))

        # MLP
        train_loader = DataLoader(
            TensorDataset(X_all[train_idx], y_all[train_idx]),
            batch_size=batch_size, shuffle=True,
        )
        model = ConnectivityMLP(n_features=X_all.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(mlp_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

        metrics = _eval_torch(model, X_all[test_idx], y_all[test_idx], device, batch_size)
        mlp_metrics.append(metrics)
        print(f"  Fold {fold+1}: RF AUC={rf_aucs[-1]:.3f}  MLP AUC={metrics['auc']:.3f}")

    rf_summary = {"rf_auc_mean": float(np.mean(rf_aucs)), "rf_auc_std": float(np.std(rf_aucs))}
    mlp_summary = _summarize_and_log(mlp_metrics, {"representation": "connectivity", "model": "mlp"}, output_dir)
    print(f"RF:  {rf_summary['rf_auc_mean']:.3f} ± {rf_summary['rf_auc_std']:.3f}")
    return {**rf_summary, **mlp_summary}


# ---------------------------------------------------------------------------
# Raw + 1D CNN
# ---------------------------------------------------------------------------

def cross_val_raw_cnn(
    features_dir: str,
    subjects: List[int],
    channels: Optional[List[int]] = None,
    n_per_subject: int = 32,
    n_splits: int = 5,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
    output_dir: str = "outputs/results/raw_cnn",
) -> Dict:
    from src.data.datasets import RawDataset
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds = RawDataset(features_dir, subjects=subjects, diagnosis_filter=["AD", "HC"], channels=channels)
    X_all, y_all, subject_ids = _smart_sample(ds, n_per_subject)
    print(f"Raw CNN: X={X_all.shape}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        train_loader = DataLoader(
            TensorDataset(X_all[train_idx], y_all[train_idx]),
            batch_size=batch_size, shuffle=True,
        )
        model = Simple1DCNN(n_channels=X_all.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

        metrics = _eval_torch(model, X_all[test_idx], y_all[test_idx], device, batch_size)
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: AUC={metrics['auc']:.3f}")

    return _summarize_and_log(fold_metrics, {"representation": "raw", "model": "cnn1d"}, output_dir)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _eval_torch(model, X_test, y_test, device, batch_size):
    model.eval()
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(yb.numpy())
    return compute_metrics(np.array(labels), np.array(probs), np.array(preds))


def _summarize_and_log(fold_metrics: list, config: dict, output_dir: str) -> Dict:
    df = pd.DataFrame(fold_metrics)
    summary = {
        f"{col}_mean": float(df[col].mean())
        for col in ["auc", "f1", "sensitivity", "specificity"]
    }
    summary.update({
        f"{col}_std": float(df[col].std())
        for col in ["auc", "f1", "sensitivity", "specificity"]
    })
    print(f"\nCV Results: AUC={summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
    log_experiment(config, summary, output_dir)
    return summary