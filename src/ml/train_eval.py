import sys
import os
import numpy as np
import pandas as pd
import datetime
import tqdm
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from src.ml.utils import compute_metrics
from src.ml.datasets import RawDataset
from src.ml.models import Simple1DCNN

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from src.ml.utils import set_seed, compute_metrics, log_experiment
from src.ml.datasets import BandpowerDataset, bandpower_to_numpy


def build_bandpower_arrays(results_root: str, subjects: List[int], channels=None):
    """Load X, y and subject_ids aligned per-window."""
    # Load windows + labels
    X, y = bandpower_to_numpy(
        results_root=results_root,
        subjects=subjects,
        diagnosis_filter=['AD', 'HC'],
        channels=channels if channels is not None else 'all'
    )

    # Build subject_ids per window, using BandpowerDataset metadata
    ds = BandpowerDataset(results_root, subjects=subjects, diagnosis_filter=['AD', 'HC'], 
        channels=channels if channels is not None else 'all')
    subject_ids = np.zeros(len(ds), dtype=int)

    for i in range(len(ds)):
        cumulative = 0
        for sample in ds.samples:
            n_win = sample['n_windows']
            if cumulative + n_win > i:
                subject_ids[i] = sample['subject_id']
                break
            cumulative += n_win

    return X, y, subject_ids

def cross_val_raw_1dcnn(
    results_root: str, subjects: List[int], channels=None, n_samples_per_subject=32
):
    """Sample 32 windows/subject → 65×32 = 2,080 total (42x reduction!)"""
    set_seed(42)
    device = "cpu"
    
    # Collect 32 random windows PER SUBJECT (much smarter)
    all_X, all_y, all_subject_ids = [], [], []
    
    dataset = RawDataset(results_root, subjects=subjects, diagnosis_filter=['AD', 'HC'], channels=channels)
    
    for sample in dataset.samples:
        # Load this subject's ALL windows once
        tensor = torch.load(sample['path'])  # [n_windows, 19, 1000]
        
        # Sample 32 random windows (or all if <32)
        n_win = min(32, tensor.shape[0])
        indices = torch.randperm(tensor.shape[0])[:n_win]
        
        subject_windows = tensor[indices, channels or slice(None), :]  # [32, 19, 1000]
        all_X.append(subject_windows)
        all_y.extend([0 if sample['diagnosis']=='AD' else 1] * n_win)
        all_subject_ids.extend([sample['subject_id']] * n_win)
    
    X_all = torch.cat(all_X)  # [2080, 19, 1000]
    y_all = torch.tensor(all_y)
    subject_ids = np.array(all_subject_ids)
    
    print(f"✅ Smart sampling: {X_all.shape[0]} windows (32/subject)")
    
    # Now GroupKFold + train FAST
    gkf = GroupKFold(n_splits=5)
    auc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        X_train = X_all[train_idx].to(device)
        y_train = y_all[train_idx]
        X_test = X_all[test_idx].to(device)
        y_test = y_all[test_idx]
        
        model = Simple1DCNN(n_channels=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 5 epochs, tiny dataset = 30 seconds
        for epoch in range(5):
            model.train()
            for i in range(0, len(X_train), 128):
                batch_X = X_train[i:i+128]
                batch_y = y_train[i:i+128]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            probs = torch.softmax(outputs, 1)[:, 1]
            auc = roc_auc_score(y_test.cpu(), probs.cpu())
            auc_scores.append(auc)
    
    print(f"🏆 Raw + 1D CNN: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    return np.mean(auc_scores)

def cross_val_bandpower_rf(
    results_root: str,
    subjects: List[int],
    rf_params: Dict[str, Any],
    channels: List[int] = None,
    n_splits: int = 5,
    save_logs: bool = True,
    output_dir: str = "results/ml_bandpower_rf"
):
    set_seed(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("🔍 Loading bandpower arrays...")
    X, y, subject_ids = build_bandpower_arrays(results_root, subjects, channels=channels)
    print(f"X shape: {X.shape}, y: {np.bincount(y)}")

    gkf = GroupKFold(n_splits=n_splits)
    auc_scores, f1_scores = [], []

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subject_ids)):
        print(f"\n📂 Fold {fold+1}/{n_splits}: {len(train_idx)} train, {len(test_idx)} test windows")

        rf = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", None),
            min_samples_split=rf_params.get("min_samples_split", 2),
            random_state=42 + fold,
            n_jobs=-1
        )

        rf.fit(X[train_idx], y[train_idx])

        y_prob = rf.predict_proba(X[test_idx])[:, 1]
        y_pred = rf.predict(X[test_idx])

        metrics = compute_metrics(y[test_idx], y_prob, y_pred)
        auc_scores.append(metrics['auc'])
        f1_scores.append(metrics['f1'])

        print(f"  AUC: {metrics['auc']:.3f}, F1: {metrics['f1']:.3f}")
        fold_results.append(metrics)

    # Aggregate
    auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)

    print("\n📊 FINAL CV RESULTS (Bandpower + RF):")
    print(f"AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"F1:  {f1_mean:.3f} ± {f1_std:.3f}")

    # Optional: log
    if save_logs:
        config = {
            "representation": "bandpower",
            "model": "RandomForest",
            **rf_params,
            "n_splits": n_splits
        }
        metrics_summary = {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std
        }
        log_experiment(config, metrics_summary, output_dir)

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std
    }

def train_final_model_bandpower_rf(results_root, subjects, rf_params, output_dir):
    """Retrain on FULL dataset after CV hyperparam selection"""
    set_seed(42)
    
    print("🎯 Training FINAL model on full dataset...")
    X, y, _ = build_bandpower_arrays(results_root, subjects)
    
    rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Channel importances
    channel_importances = rf.feature_importances_.reshape(19, 5).mean(1)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / f"final_model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model + metadata
    import joblib
    joblib.dump(rf, model_dir / "model.joblib")
    
    metadata = {
        'rf_params': rf_params,
        'n_train_samples': X.shape[0],
        'n_features': X.shape[1],
        'channel_importances': channel_importances,
        'subjects_used': subjects
    }
    pd.DataFrame([metadata]).to_csv(model_dir / "metadata.csv", index=False)
    
    print(f"💾 FINAL MODEL SAVED: {model_dir / 'model.joblib'}")
    print(f"🏆 Top 5 channels: {np.argsort(channel_importances)[-5:][::-1]}")
    
    return rf, model_dir

def cross_val_stft_2dcnn(
    results_root: str,
    subjects: List[int],
    channels: Optional[List[int]] = None,
    n_samples_per_subject: int = 32,
    n_splits: int = 5,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    device: str = None,
    output_dir: str = "results/ml_stft_2dcnn"
):
    """Subject-wise CV for STFT spectrograms + 2D CNN (32 windows/subject)"""
    from src.ml.utils import set_seed
    set_seed(42)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # STEP 1: SMART SAMPLING (32 windows/subject = 2K total)
    print("🧠 Smart sampling: 32 windows per subject...")
    from src.ml.datasets import STFTDataset
    dataset = STFTDataset(results_root, subjects=subjects, diagnosis_filter=['AD', 'HC'], channels=channels)
    
    all_X, all_y, all_subject_ids = [], [], []
    
    for sample in dataset.samples:
        tensor = torch.load(sample['path'])  # [n_windows, 19, 129, 7]
        n_win = min(n_samples_per_subject, tensor.shape[0])
        
        # Use FIRST n_win windows (more stable than random)
        indices = torch.arange(n_win)
        ch_slice = slice(None) if channels is None else torch.tensor(channels)
        subject_windows = tensor[indices][:, ch_slice, :, :]  # [32, n_ch, 129, 7]
        
        all_X.append(subject_windows)
        label = 0 if sample['diagnosis'] == 'AD' else 1
        all_y.extend([label] * n_win)
        all_subject_ids.extend([sample['subject_id']] * n_win)
    
    X_all = torch.cat(all_X)  # [2080, n_ch, 129, 7]
    y_all = torch.tensor(all_y)
    subject_ids = np.array(all_subject_ids)
    
    print(f"✅ Loaded: X={X_all.shape}, y={y_all.shape}, {len(np.unique(subject_ids))} subjects")
    
    # STEP 2: GroupKFold splits
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    auc_scores, f1_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        print(f"\n📂 Fold {fold+1}/{n_splits}: {len(train_idx)} train, {len(test_idx)} test windows")
        
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        # In-memory TensorDatasets (blazing fast)
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # STEP 3: Train 2D CNN
        from src.ml.models import Simple2DCNN
        model = Simple2DCNN(n_channels=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            train_loss = 0
            from tqdm import tqdm
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # STEP 4: Evaluate
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        
        from src.ml.utils import compute_metrics
        metrics = compute_metrics(np.array(all_labels), np.array(all_probs), np.array(all_preds))
        auc_scores.append(metrics['auc'])
        f1_scores.append(metrics['f1'])
        
        print(f"  AUC: {metrics['auc']:.3f}, F1: {metrics['f1']:.3f}")
    
    # STEP 5: Aggregate + log
    auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    print(f"\n📊 FINAL CV RESULTS (STFT + 2D CNN):")
    print(f"AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    
    if len(auc_scores) > 1:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(auc_scores, 0.5)
        print(f"p-value vs random: {p_value:.2e}")
    
    config = {
        "representation": "stft",
        "model": "Simple2DCNN",
        "n_channels": len(channels) if channels else 19,
        "n_samples_per_subject": n_samples_per_subject,
        "n_splits": n_splits
    }
    metrics_summary = {"auc_mean": auc_mean, "auc_std": auc_std}
    from src.ml.utils import log_experiment
    log_experiment(config, metrics_summary, output_dir)
    
    return {"auc_mean": auc_mean, "auc_std": auc_std}

def cross_val_connectivity(
    results_root: str,
    subjects: List[int],
    n_samples_per_subject: int = 32,
    n_splits: int = 5,
    mlp_epochs: int = 50,
    mlp_batch_size: int = 64,
    mlp_lr: float = 1e-3,
    rf_n_estimators: int = 200,
    device: str = None,
    output_dir: str = "results/ml_connectivity"
):
    """Benchmark ConnectivityMLP vs RF baseline on PLV pairs"""
    from src.ml.utils import set_seed
    set_seed(42)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Smart sampling (32 windows/subject)
    print("🧠 Smart sampling connectivity: 32 windows/subject...")
    from src.ml.datasets import ConnectivityDataset
    dataset = ConnectivityDataset(results_root, subjects=subjects, diagnosis_filter=['AD', 'HC'])
    
    all_X, all_y, all_subject_ids = [], [], []
    
    for sample in dataset.samples:
        tensor = torch.load(sample['path'])  # [n_windows, 171]
        n_win = min(n_samples_per_subject, tensor.shape[0])
        
        # First n_win windows
        indices = torch.arange(n_win)
        subject_windows = tensor[indices]  # [32, 171]
        
        all_X.append(subject_windows)
        label = 0 if sample['diagnosis'] == 'AD' else 1
        all_y.extend([label] * n_win)
        all_subject_ids.extend([sample['subject_id']] * n_win)
    
    X_all = torch.cat(all_X)  # [2080, 171]
    y_all = torch.tensor(all_y)
    subject_ids = np.array(all_subject_ids)
    
    print(f"✅ Loaded: X={X_all.shape}, y={y_all.shape}, {len(np.unique(subject_ids))} subjects")
    
    # STEP 2: GroupKFold splits (same as bandpower/RF!)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # RF BASELINE (like bandpower)
    print("\n=== RF BASELINE ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    rf_auc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        print(f"RF Fold {fold+1}/{n_splits}")
        
        X_train_rf = X_all[train_idx].numpy()
        y_train_rf = y_all[train_idx].numpy()
        X_test_rf = X_all[test_idx].numpy()
        y_test_rf = y_all[test_idx].numpy()
        
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=42+fold, n_jobs=-1
        )
        rf.fit(X_train_rf, y_train_rf)
        y_prob = rf.predict_proba(X_test_rf)[:, 1]
        rf_auc_scores.append(roc_auc_score(y_test_rf, y_prob))
    
    rf_auc_mean = np.mean(rf_auc_scores)
    print(f"🏆 Connectivity + RF: {rf_auc_mean:.3f} ± {np.std(rf_auc_scores):.3f}")
    
    # MLP BENCHMARK
    print("\n=== CONNECTIVITYMLP ===")
    from src.ml.models import ConnectivityMLP
    from src.ml.utils import compute_metrics
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    
    mlp_auc_scores, mlp_f1_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=subject_ids)):
        print(f"MLP Fold {fold+1}/{n_splits}")
        
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=mlp_batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=mlp_batch_size, shuffle=False, num_workers=0)
        
        # Train MLP
        model = ConnectivityMLP(n_features=171).to(device)
        optimizer = optim.Adam(model.parameters(), lr=mlp_lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(mlp_epochs):
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss={epoch_loss/len(train_loader):.3f}")
        
        # Evaluate
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        
        metrics = compute_metrics(np.array(all_labels), np.array(all_probs), np.array(all_preds))
        mlp_auc_scores.append(metrics['auc'])
        mlp_f1_scores.append(metrics['f1'])
        print(f"  AUC: {metrics['auc']:.3f}, F1: {metrics['f1']:.3f}")
    
    # FINAL RESULTS
    print(f"\n📊 CONNECTIVITY COMPARISON:")
    print(f"RF:           {rf_auc_mean:.3f} ± {np.std(rf_auc_scores):.3f}")
    print(f"ConnectivityMLP: {np.mean(mlp_auc_scores):.3f} ± {np.std(mlp_auc_scores):.3f}")
    
    # Log both
    from src.ml.utils import log_experiment
    log_experiment({
        "representation": "connectivity", "model": "RandomForest", 
        "n_samples_per_subject": n_samples_per_subject
    }, {"auc_mean": rf_auc_mean}, f"{output_dir}")
    
    log_experiment({
        "representation": "connectivity", "model": "ConnectivityMLP",
        "n_samples_per_subject": n_samples_per_subject
    }, {"auc_mean": np.mean(mlp_auc_scores)}, f"{output_dir}")
    
    return {
        "rf_auc": rf_auc_mean,
        "mlp_auc": np.mean(mlp_auc_scores),
        "rf_std": np.std(rf_auc_scores),
        "mlp_std": np.std(mlp_auc_scores)
    }
 
if __name__ == "__main__":
    conn_root = "results/connectivity_20260202_2157"
    subjects = list(range(1, 66))
    
    print("=== CONNECTIVITY BENCHMARK ===")
    results = cross_val_connectivity(conn_root, subjects)
    print(f"RF:  {results['rf_auc']:.3f}, MLP: {results['mlp_auc']:.3f}")
