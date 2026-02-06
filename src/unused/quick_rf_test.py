import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
from pathlib import Path

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.ml.utils import compute_metrics, set_seed
from src.ml.datasets import BandpowerDataset, bandpower_to_numpy

if __name__ == "__main__":
    set_seed(42)
    
    print("🔍 Loading FULL dataset (subjects 1-65)...")
    X, y = bandpower_to_numpy(
        'results/bandpower_20260202_2134',
        subjects=list(range(1, 66)),
        diagnosis_filter=['AD', 'HC']
    )
    print(f"Total data: X={X.shape}, y={np.bincount(y)}")
    
    # Build subject IDs array (each window knows its subject)
    print("🔍 Building subject mapping...")
    ds = BandpowerDataset('results/bandpower_20260202_2134', subjects=list(range(1, 66)))
    subject_ids = np.zeros(len(ds))
    for i in range(len(ds)):
        # Get subject from dataset logic
        cumulative = 0
        for sample in ds.samples:
            n_win = sample['n_windows']
            if cumulative + n_win > i:
                subject_ids[i] = sample['subject_id']
                break
            cumulative += n_win
    
    print(f"Subject distribution: {np.bincount(subject_ids.astype(int))}")
    
    # 5-FOLD SUBJECT-WISE CV
    print("\n🚀 5-FOLD SUBJECT CV (NO LEAKAGE!)")
    gkf = GroupKFold(n_splits=5)
    auc_scores, f1_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subject_ids)):
        print(f"Fold {fold+1}/5: {len(train_idx)} train, {len(test_idx)} test windows")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42+fold, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        
        y_prob = rf.predict_proba(X[test_idx])[:, 1]
        y_pred = rf.predict(X[test_idx])
        
        metrics = compute_metrics(y[test_idx], y_prob, y_pred)
        auc_scores.append(metrics['auc'])
        f1_scores.append(metrics['f1'])
        
        print(f"  AUC: {metrics['auc']:.3f}, F1: {metrics['f1']:.3f}")
    
    # Results
    print("\n📊 FINAL RESULTS:")
    print(f"5-fold CV AUC:  {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"5-fold CV F1:   {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(auc_scores, 0.5)
    print(f"p-value vs random: {p_val:.1e} (p<0.05 = significant)")
    
    # BEST channel set from averaged importances
    all_importances = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subject_ids)):
        rf = RandomForestClassifier(n_estimators=100, random_state=42+fold)
        rf.fit(X[train_idx], y[train_idx])
        imp = rf.feature_importances_.reshape(19, 5).mean(1)
        all_importances.append(imp)
    
    avg_importance = np.mean(all_importances, 0)
    top_channels = np.argsort(avg_importance)[-5:]
    print(f"\n🏆 Top 5 channels (CV stable): {top_channels}")
    