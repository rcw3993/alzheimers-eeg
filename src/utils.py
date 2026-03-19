import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from typing import Dict


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def diagnosis_to_label(diagnosis: str) -> int:
    return {"AD": 0, "HC": 1, "FTD": 2}[diagnosis]


def compute_metrics(y_true, y_prob, y_pred=None) -> Dict:
    auc = roc_auc_score(y_true, y_prob)
    if y_pred is not None:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        sensitivity = report.get("0", {}).get("recall", float("nan"))
        specificity = report.get("1", {}).get("recall", float("nan"))
    else:
        f1 = sensitivity = specificity = float("nan")
    return {"auc": auc, "f1": f1, "sensitivity": sensitivity, "specificity": specificity}


def ttest_vs_chance(auc_scores) -> Dict:
    t, p = stats.ttest_1samp(auc_scores, 0.5)
    return {"mean_auc": float(np.mean(auc_scores)), "std_auc": float(np.std(auc_scores)), "p_value": float(p)}


def log_experiment(config: dict, metrics: dict, output_dir: str) -> Path:
    """Append a row to experiment_logs.csv in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = {**config, **metrics, "timestamp": datetime.now().isoformat()}
    log_path = output_dir / "experiment_logs.csv"

    if log_path.exists():
        df = pd.concat([pd.read_csv(log_path), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_path, index=False)
    print(f"Logged to {log_path}")
    return log_path