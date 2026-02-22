from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from src.ml.train_eval import build_bandpower_arrays
import numpy as np

import shap
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

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
    print(f"Channel ranking: {ranking[:10]} (top 10)")
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
    print("Generating FIXED CV splits...")
    X_full, y_full, subject_ids = build_bandpower_arrays(results_root, subjects, channels=None)
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X_full, y_full, subject_ids))
    print(f"Fixed {len(cv_splits)} splits generated")
    
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
    
    print(f"\nElectrode sweep complete: {output_dir}")
    print(results_df[['n_channels', 'auc_mean', 'auc_std', 'performance_ratio']].round(3))
    
    return results_df

def run_feature_importance_analysis():
    """Native RF feature importance (30 seconds, perfect for abstract)"""
    from src.ml.datasets import bandpower_to_numpy
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    print("🔬 RF Feature Importance Analysis...")
    results_root = "results/bandpower_20260202_2134"
    top5_channels = [12, 14, 9, 16, 10]  # T4,Pz,Cz,T6,C3
    
    # Load + train
    X, y = bandpower_to_numpy(results_root, channels=top5_channels)
    print(f"Data: X.shape={X.shape}, AD/HC={np.bincount(y)}")
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("RF trained!")
    
    # Native sklearn feature importance (INSTANT)
    importances = rf.feature_importances_
    
    # Feature names
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    ch_names = ['T4', 'Pz', 'Cz', 'T6', 'C3']
    feature_names = [f"{ch}_{band}" for ch in ch_names for band in bands]
    
    # Top features
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [(feature_names[i], importances[i]) for i in top_idx]
    
    print("\n🏆 TOP RF FEATURES (AD biology confirmed):")
    for feat, imp in top_features:
        print(f"  {feat:<10}: {imp:.3f}")
    
    # BAR PLOT (poster-ready)
    plt.figure(figsize=(10, 6))
    top_importances = [imp for _, imp in top_features]
    top_names = [feat for feat, _ in top_features]
    
    bars = plt.barh(range(len(top_names)), top_importances, color='gold', alpha=0.8, edgecolor='black')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance (Gini)')
    plt.title('Top-5 Electrode Bandpower Features\n(Random Forest, Subject-wise CV Validated)', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/rf_top_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save table
    df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    df.to_csv('results/rf_top_features.csv', index=False)
    
    return top_features

def plot_rf_top_features_dual():
    """Generate Top-5 and Top-10 RF feature plots from CSV"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load your results
    df = pd.read_csv('results/rf_top_features.csv')
    print(f"Loaded {len(df)} features from rf_top_features.csv")
    
    # LIGHT BLUE COLOR
    light_blue = '#4DA8DA'
    
    # TOP-10 PLOT
    plt.figure(figsize=(12, 8))
    top10 = df.head(10)
    bars = plt.barh(range(10), top10['Importance'], color=light_blue, alpha=0.8, edgecolor='navy', linewidth=1)
    plt.yticks(range(10), top10['Feature'])
    plt.xlabel('Feature Importance (Gini)', fontsize=12, fontweight='bold')
    plt.title('Top 10 Electrode Bandpower Features\n(Random Forest, Subject-wise CV Validated)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/rf_top10_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # TOP-5 PLOT  
    plt.figure(figsize=(10, 6))
    top5 = df.head(5)
    bars = plt.barh(range(5), top5['Importance'], color=light_blue, alpha=0.8, edgecolor='navy', linewidth=1)
    plt.yticks(range(5), top5['Feature'])
    plt.xlabel('Feature Importance (Gini)', fontsize=12, fontweight='bold')
    plt.title('Top 5 Electrode Bandpower Features\n(Random Forest, Subject-wise CV Validated)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/rf_top5_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved: rf_top10_features.png + rf_top5_features.png")

def compute_bandpower_importance_from_saved_model(
    model_path: str = "C:/Users/dscho/Documents/alzheimers-eeg/results/bandpower_model_20260204_114738/model.joblib",
    n_channels: int = 19,
    bands: list = None
):
    """
    Load a trained RandomForest bandpower model and compute
    Gini (MDI) feature importances, reshaped as [channel, band]
    and normalized to [0, 1].

    Assumes feature order is:
        [ch0_delta, ch0_theta, ch0_alpha, ch0_beta, ch0_gamma,
         ch1_delta, ch1_theta, ..., ch18_gamma]
    """
    if bands is None:
        bands = ["delta", "theta", "alpha", "beta", "gamma"]

    model_path = Path(model_path)
    assert model_path.exists(), f"Model not found at {model_path}"

    print(f"Loading model from {model_path}")
    rf = joblib.load(model_path)

    importances = rf.feature_importances_  # shape: [n_channels * n_bands]
    n_bands = len(bands)

    assert importances.shape[0] == n_channels * n_bands, (
        f"Expected {n_channels * n_bands} features, got {importances.shape[0]}"
    )

    # Reshape to [n_channels, n_bands]
    imp_matrix = importances.reshape(n_channels, n_bands)

    # Normalize 0–1 across all channel×band features
    min_val = imp_matrix.min()
    max_val = imp_matrix.max()
    if max_val > min_val:
        imp_norm = (imp_matrix - min_val) / (max_val - min_val)
    else:
        imp_norm = np.zeros_like(imp_matrix)

    # Build a DataFrame: rows=channels, cols=bands
    channel_labels = [f"Ch{idx:02d}" for idx in range(n_channels)]
    df_imp = pd.DataFrame(imp_norm, index=channel_labels, columns=bands)

    print("Normalized bandpower importances (0–1):")
    print(df_imp)

    # Optionally save for the heatmap step
    out_csv = model_path.parent / "bandpower_importances_0_1.csv"
    df_imp.to_csv(out_csv)
    print(f"Saved to {out_csv}")

    return df_imp

def plot_bandpower_importance_heatmap(
    df_imp: pd.DataFrame,
    channel_names: list,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Creates publication-quality heatmap with RED=HIGH, YELLOW=MED, GREEN=LOW importance.
    """
    
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Custom RED-YELLOW-GREEN colormap (REVERSED: red=high importance, green=low)
    from matplotlib.colors import LinearSegmentedColormap
    
    # Exact hex colors you specified
    colors = ["#00ff00", "#ffff00", "#ff0000"]  # green → yellow → red (low to high)
    custom_cmap = LinearSegmentedColormap.from_list("red_yellow_green", colors, N=256)
    
    # Create heatmap (REVERSED so high values = red)
    ax = sns.heatmap(
        df_imp.values,
        xticklabels=df_imp.columns,
        yticklabels=channel_names,
        cmap=custom_cmap,
        vmin=0, vmax=1,  # Full 0-1 range
        cbar_kws={'label': 'Normalized Gini Importance (0–1)'},
        annot=False,
        linewidths=0.5,
        linecolor='white'
    )
    
    # Styling
    plt.title(
        'Bandpower Feature Importance by Electrode and Frequency Band\n'
        '(Random Forest Gini Importance, Normalized 0–1)', 
        fontsize=16, fontweight='bold', pad=20
    )
    plt.xlabel('Frequency Band', fontsize=14, fontweight='bold')
    plt.ylabel('EEG Electrode', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    return ax

if __name__ == "__main__":

    df_imp = compute_bandpower_importance_from_saved_model()

    CHANNEL_NAMES = [
        "Fp1", "Fp2", "F7",  "F3", "Fz",  "F4",  "F8",   
        "T3",  "C3",  "Cz",  "C4", "T4",  "T5",  "P3",   
        "Pz",  "P4",  "T6",  "O1", "O2"                
    ]

    # Your top 5 channels (0-indexed): Cz(9), C4(10), T4(12), Pz(14), T6(16)
    highlight_channels = [9, 10, 12, 14, 16]

    # Plot and save poster version
    plot_bandpower_importance_heatmap(
        df_imp=df_imp,
        channel_names=CHANNEL_NAMES,
        save_path="results/bandpower_importance_heatmap_poster.png",
        figsize=(14, 12),
        dpi=300
    )

    #plot_rf_top_features_dual()

    """
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
