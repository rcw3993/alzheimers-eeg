import sys
import os
from pathlib import Path

# Add PROJECT ROOT to path (same as pipeline.py)
project_root = Path(__file__).parent.parent.parent  # ml/ → src/ → root
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# Local imports (project root in path)
from datasets import BandpowerDataset, bandpower_to_numpy
from utils import set_seed

set_seed(42)

# TEST 1: PyTorch Dataset (for CNN/MLP)
print("=== TEST 1: PyTorch Dataset ===")
ds = BandpowerDataset(
    results_root='results/bandpower_20260202_2134',  # UPDATE TO YOUR FOLDER
    subjects=[1, 40],  # AD + HC
    diagnosis_filter=['AD', 'HC']
)

print(f"Dataset size: {len(ds)} windows")
x, y = ds[0]
print(f"Sample X shape: {x.shape}")  # torch.Size([19, 5])
print(f"Sample y: {y}")              # tensor(0) = AD

# Test DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True)
batch_x, batch_y = next(iter(loader))
print(f"Batch shape: {batch_x.shape}")  # [4, 19, 5]

# TEST 2: Numpy arrays (for RF/SVM)
print("\n=== TEST 2: Sklearn Arrays ===")
X, y = bandpower_to_numpy(
    results_root='results/bandpower_20260202_2134',  # UPDATE
    subjects=[1, 40]
)
print(f"X shape: {X.shape}")  # [n_windows, 95] = 19ch × 5bands
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y, return_counts=True)}")  # AD=0, HC=1

# TEST 3: Electrode reduction
print("\n=== TEST 3: Minimal Electrodes ===")
ds_reduced = BandpowerDataset(
    results_root='results/bandpower_20260202_2134',
    subjects=[1, 40],
    channels=[0, 5, 12]  # Example: 3 channels
)
x_red, y_red = ds_reduced[0]
print(f"Reduced X shape: {x_red.shape}")  # [3, 5]

print("\n✅ ALL TESTS PASSED!")
