# src/ml/utils.py
import numpy as np
import torch
from scipy import stats
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from typing import Dict, Any, Tuple

def set_seed(seed=42):
    """Ensure reproducible results across runs"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set: {seed}")

def diagnosis_to_label(diagnosis: str) -> int:
    """AD=0, HC=1, FTD=2 (binary: AD vs non-AD possible too)"""
    mapping = {"AD": 0, "HC": 1, "FTD": 2}
    return mapping[diagnosis]

def compute_metrics(y_true, y_prob, y_pred=None):
    """Full metrics suite with statistical rigor"""
    auc = roc_auc_score(y_true, y_prob)
    
    if y_pred is not None:
        f1 = f1_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        sensitivity = report['0']['recall']  # AD sensitivity
        specificity = report['1']['recall']  # HC specificity
    else:
        f1 = sensitivity = specificity = np.nan
    
    return {
        'auc': auc, 'f1': f1, 'sensitivity': sensitivity, 'specificity': specificity
    }

def paired_ttest(results_list):
    """Statistical significance: p-value across folds"""
    aucs = [r['auc'] for r in results_list]
    t_stat, p_value = stats.ttest_1samp(aucs, 0.5)  # vs random classifier
    return {'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs), 'p_value': p_value}

def log_experiment(config: dict, metrics: dict, output_dir: str):
    """Save results to CSV + JSON for tables/leaderboards"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_df = pd.DataFrame([{
        **config, **metrics, 'timestamp': timestamp
    }])
    
    log_path = Path(output_dir) / 'experiment_logs.csv'
    if log_path.exists():
        existing_df = pd.read_csv(log_path)
        log_df = pd.concat([existing_df, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    
    # Summary table (FIXED - handle missing columns)
    available_cols = [col for col in ['auc', 'train_time'] if col in log_df.columns]
    if len(available_cols) > 0:
        summary_df = log_df.groupby('representation')[available_cols].agg(['mean', 'std']).round(3)
        summary_df.to_csv(Path(output_dir) / 'summary_table.csv')
    
    print(f"Logged: {log_path}")
    return log_path
