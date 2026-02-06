from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from src.ml.train_eval import build_bandpower_arrays
import numpy as np

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.ml.utils import set_seed
from src.ml.train_eval import cross_val_bandpower_rf, train_final_model_bandpower_rf
from src.ml.datasets import bandpower_to_numpy
import joblib

def extract_channel_ranking(model_path: str) -> np.ndarray:
    """Extract stable channel ranking from final trained model"""
    model = joblib.load(model_path)
    if hasattr(model, 'channel_importances_'):
        importances = model.channel_importances_
    else:
        # Fallback: compute from feature_importances_
        importances = model.feature_importances_.reshape(19, 5).mean(1)
    
    ranking = np.argsort(importances)[::-1]  # Highest importance first
    print(f"🏆 Channel ranking: {ranking[:10]} (top 10)")
    return ranking

def electrode_sweep_bandpower(
    results_root: str,
    subjects: List[int],
    channel_ranking: np.ndarray,
    top_k_list: List[int] = [1, 3, 5, 8, 12, 19],
    rf_params: Dict = None,
    output_dir: str = None
):
    """Test performance vs #electrodes using FIXED CV SPLITS across all k"""
    from src.ml.utils import set_seed
    set_seed(42)

    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/electrode_sweep_bandpower_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Generate FIXED CV splits ONCE (using full 19ch data)
    print("🔍 Generating FIXED CV splits...")
    X_full, y_full, subject_ids = build_bandpower_arrays(results_root, subjects, channels=None)
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X_full, y_full, subject_ids))
    print(f"✅ Fixed {len(cv_splits)} splits generated")
    
    results = {}
    
    for k in top_k_list:
        channels = channel_ranking[:k]
        print(f"\n⚡ Testing top-{k} channels: {channels}")
        
        # Load REDUCED channel data (same splits, fewer features)
        X_k, y_k, subject_ids_k = build_bandpower_arrays(results_root, subjects, channels=channels)
        
        auc_scores = []
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            print(f"  Fold {fold+1}: {len(train_idx)} train, {len(test_idx)} test")
            
            rf = RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 200) if rf_params else 200,
                max_depth=rf_params.get("max_depth", None) if rf_params else None,
                min_samples_split=rf_params.get("min_samples_split", 2) if rf_params else 2,
                random_state=42 + fold,
                n_jobs=-1
            )
            
            rf.fit(X_k[train_idx], y_k[train_idx])
            y_prob = rf.predict_proba(X_k[test_idx])[:, 1]
            auc = roc_auc_score(y_k[test_idx], y_prob)
            auc_scores.append(auc)
            print(f"    AUC: {auc:.3f}")
        
        auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
        results[f"top_{k}"] = {
            "n_channels": k,
            "channels": channels.tolist(),
            "auc_mean": auc_mean,
            "auc_std": auc_std,
        }
        print(f"  Top-{k}: {auc_mean:.3f} ± {auc_std:.3f}")
    
    # Compute ratios vs top_19 baseline
    results_df = pd.DataFrame(results).T
    baseline_auc = results_df.loc["top_19", "auc_mean"]
    results_df["performance_ratio"] = (results_df["auc_mean"] / baseline_auc).round(3)
    
    # Save + plot
    results_df.to_csv(Path(output_dir) / "electrode_performance.csv")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['n_channels'], results_df['auc_mean'], 
                results_df['auc_std'], fmt='o-', capsize=5)
    plt.axhline(baseline_auc, color='r', linestyle='--', label=f'Full 19ch ({baseline_auc:.3f})')
    plt.xlabel('# Electrodes')
    plt.ylabel('CV AUC')
    plt.title('Electrode Reduction: Bandpower + RF (Fixed Splits)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(output_dir) / "performance_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Electrode sweep complete: {output_dir}")
    print(results_df[['n_channels', 'auc_mean', 'auc_std', 'performance_ratio']].round(3))
    
    return results_df


if __name__ == "__main__":
    results_df = pd.DataFrame({
        'Representation': ['Bandpower\n(RF)', 'Connectivity\n(RF)', 'STFT\n(2D CNN)', 'Raw\n(1D CNN)'],
        'AUC': [0.815, 0.769, 0.747, 0.508],
        'Features': [95, 171, 17157, 19000],
        'Electrodes': [5, 'N/A', 19, 19]
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#A9A9A9']

    # LEFT: AUC (your code - perfect)
    bars = ax1.bar(results_df['Representation'], results_df['AUC'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.axhline(0.815, color='red', linestyle='--', linewidth=2, label='Bandpower Baseline')
    ax1.set_ylabel('Cross-Validation AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Representation Comparison\n(Subject-wise 5-fold CV)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.45, 0.85)
    ax1.grid(True, alpha=0.3)
    for bar, auc in zip(bars, results_df['AUC']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    # RIGHT: FIXED bar plot (log scale)
    ax2.bar(range(4), results_df['Features'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_ylabel('Number of Features\n(log₁₀ scale)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['Bandpower\n95', 'Connectivity\n171', 'STFT\n17K', 'Raw\n19K'], rotation=0)
    ax2.set_title('Compute Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    """
    # DEBUG: Find exact preprocessing folder
    import glob
    bandpower_folders = glob.glob("results/bandpower_*")
    print(f"Available bandpower folders: {bandpower_folders}")
    
    # Use FIRST one (your actual data)
    results_root = bandpower_folders[0]
    print(f"Using: {results_root}")
    
    # Find your trained model
    model_paths = list(Path("results").rglob("model.joblib"))
    model_path = model_paths[0]  # First trained model
    print(f"Using trained model: {model_path}")
    
    channel_ranking = extract_channel_ranking(model_path)
    
    subjects = list(range(1, 66))
    electrode_results = electrode_sweep_bandpower(
        results_root=results_root,  # ← FIXED PATH
        subjects=subjects,
        channel_ranking=channel_ranking,
        top_k_list=[1, 3, 5, 8, 12, 19]
    )
    """
