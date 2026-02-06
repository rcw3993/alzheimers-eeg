import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Tuple
from src.preprocessing.loaders import get_diagnosis, load_participants_tsv

class RawDataset(Dataset):
    """
    PyTorch Dataset for raw EEG windows from preprocessed results.
    Shape: (n_windows, n_channels, n_timepoints=1000) → [n_ch, 1000] for 1D CNN
    """
    def __init__(
        self,
        results_root: str,
        subjects: Optional[List[int]] = None,
        diagnosis_filter: Optional[List[str]] = None,  # ['AD', 'HC']
        channels: Union[str, List[int]] = 'all',
        window_subset: Optional[Tuple[int, int]] = None,  # (start, end) for testing
    ):
        self.results_root = Path(results_root)
        self.data_dir = self.results_root / 'data'
        self.summary_path = self.results_root / 'tables' / 'raw_summary.csv'  # Different summary filename
        
        # Parse subjects + diagnoses
        self._build_samples(subjects, diagnosis_filter)
        
        # Channel selection
        self.channels = self._parse_channels(channels)
        self.n_channels = len(self.channels)
        
        # Window subset for testing
        self.window_start, self.window_end = window_subset or (0, None)
    
    def _build_samples(self, subjects, diagnosis_filter):
        """Scan .pt files → build (file_path, subject_id, diagnosis) list"""
        self.samples = []
        
        # Load summary table if exists
        if self.summary_path.exists():
            summary_df = pd.read_csv(self.summary_path)
        else:
            summary_df = pd.DataFrame()
        
        # Find all RAW files (different naming!)
        pt_files = sorted(self.data_dir.glob('sub-*_raw.pt'))
        print(f"Found {len(pt_files)} raw files")
        
        for pt_file in pt_files:
            # Parse subject ID: sub-001_raw.pt → 1
            sub_str = pt_file.stem.split('_')[0]  # sub-001
            subject_id = int(sub_str.split('-')[1])
            
            # Subject filter
            if subjects is not None and subject_id not in subjects:
                continue
            
            # Diagnosis filter
            diagnosis = get_diagnosis(subject_id)
            if diagnosis_filter is not None and diagnosis not in diagnosis_filter:
                print(f"Skipping {subject_id} ({diagnosis})")
                continue
            
            # Get n_windows from summary or load tensor
            sub_row = summary_df[summary_df['subject_id'] == subject_id]
            if len(sub_row) > 0:
                n_windows = int(sub_row['n_windows'].iloc[0])
            else:
                # Fallback: load tensor
                tensor = torch.load(pt_file)
                n_windows = tensor.shape[0]
            
            self.samples.append({
                'path': pt_file,
                'subject_id': subject_id,
                'diagnosis': diagnosis,
                'n_windows': n_windows
            })
        
        print(f"✅ Loaded {len(self.samples)} subjects ({sum(s['n_windows'] for s in self.samples)} total windows)")
    
    def _parse_channels(self, channels):
        """Normalize 'channels' input into a list of valid integer indices."""
        # Case 1: all channels OR None
        if channels is None or (isinstance(channels, str) and channels.lower() == 'all'):
            return list(range(19))

        # Case 2: NumPy array or list
        if isinstance(channels, (np.ndarray, list)):
            channels_list = [int(ch) for ch in list(channels)]
            if not all(0 <= ch < 19 for ch in channels_list):
                raise ValueError(f"Invalid channels {channels_list}: must be integers 0–18")
            return channels_list

        # Case 3: fallback error
        raise ValueError(f"Unknown channels type: {type(channels)} ({channels})")
    
    def __len__(self):
        """Total number of windows across all subjects"""
        total = 0
        for sample in self.samples:
            n_win = sample['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            total += max(0, n_win)
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return single window: X=[n_ch, 1000], y=label"""
        # Find which subject/window (identical logic)
        cumulative = 0
        sample = None
        rel_idx = 0
        
        for s in self.samples:
            n_win = s['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            
            if cumulative + n_win > idx:
                sample = s
                rel_idx = idx - cumulative + self.window_start
                break
            cumulative += n_win
        
        if sample is None or rel_idx >= sample['n_windows']:
            raise IndexError(f"Dataset index {idx} out of range (total len={len(self)})")
        
        # Load tensor: [n_windows, 19, 1000] → [n_ch, 1000]
        tensor = torch.load(sample['path'])
        x = tensor[rel_idx, self.channels, :]  # [n_selected_ch, 1000]
        
        # Label (binary AD vs non-AD)
        diagnosis = sample['diagnosis']
        label = 0 if diagnosis == 'AD' else 1
        
        return x.float(), torch.tensor(label, dtype=torch.long)

class BandpowerDataset(Dataset):
    """
    PyTorch Dataset for bandpower features from preprocessed results.
    Shape: (n_windows, n_channels, n_bands=5)
    """
    def __init__(
        self,
        results_root: str,
        subjects: Optional[List[int]] = None,
        diagnosis_filter: Optional[List[str]] = None,  # ['AD', 'HC']
        channels: Union[str, List[int]] = 'all',
        window_subset: Optional[Tuple[int, int]] = None,  # (start, end) for testing
    ):
        """
        Args:
            results_root: 'results/bandpower_20260202_2105'
            subjects: [1,2,3,...] or None=load all available
            diagnosis_filter: ['AD', 'HC'] or None=keep all
            channels: 'all', list of indices [0,5,12], or 'top5'
            window_subset: (0, 1000) = first 1000 windows only (for testing)
        """
        self.results_root = Path(results_root)
        self.data_dir = self.results_root / 'data'
        self.summary_path = self.results_root / 'tables' / 'bandpower_summary.csv'
        
        # Parse subjects + diagnoses
        self._build_samples(subjects, diagnosis_filter)
        
        # Channel selection
        self.channels = self._parse_channels(channels)
        self.n_channels = len(self.channels)
        
        # Window subset for testing
        self.window_start, self.window_end = window_subset or (0, None)
    
    def _build_samples(self, subjects, diagnosis_filter):
        """Scan .pt files → build (file_path, subject_id, diagnosis) list"""
        self.samples = []
        
        # Load summary table if exists
        if self.summary_path.exists():
            summary_df = pd.read_csv(self.summary_path)
        else:
            summary_df = pd.DataFrame()
        
        # Find all bandpower files
        pt_files = sorted(self.data_dir.glob('sub-*_bandpower.pt'))
        print(f"Found {len(pt_files)} bandpower files")
        
        for pt_file in pt_files:
            # Parse subject ID: sub-001_bandpower.pt → 1
            sub_str = pt_file.stem.split('_')[0]  # sub-001
            subject_id = int(sub_str.split('-')[1])
            
            # Subject filter
            if subjects is not None and subject_id not in subjects:
                continue
            
            # Diagnosis filter
            diagnosis = get_diagnosis(subject_id)
            if diagnosis_filter is not None and diagnosis not in diagnosis_filter:
                print(f"Skipping {subject_id} ({diagnosis})")
                continue
            
            # Get n_windows from summary or load tensor
            sub_row = summary_df[summary_df['subject_id'] == subject_id]
            if len(sub_row) > 0:
                n_windows = int(sub_row['n_windows'].iloc[0])
            else:
                # Fallback: load tensor
                tensor = torch.load(pt_file)
                n_windows = tensor.shape[0]
            
            self.samples.append({
                'path': pt_file,
                'subject_id': subject_id,
                'diagnosis': diagnosis,
                'n_windows': n_windows
            })
        
        print(f"✅ Loaded {len(self.samples)} subjects ({sum(s['n_windows'] for s in self.samples)} total windows)")
    
    def _parse_channels(self, channels):
        """Normalize 'channels' input into a list of valid integer indices."""
        # Case 1: all channels
        if isinstance(channels, str) and channels.lower() == 'all':
            return list(range(19))

        # Case 2: NumPy array or list
        if isinstance(channels, (np.ndarray, list)):
            # Convert any NumPy integer types → plain Python ints
            channels_list = [int(ch) for ch in list(channels)]
            if not all(0 <= ch < 19 for ch in channels_list):
                raise ValueError(f"Invalid channels {channels_list}: must be integers 0–18")
            return channels_list

        # Case 3: fallback error
        raise ValueError(f"Unknown channels type: {type(channels)} ({channels})")
    
    def __len__(self):
        """Total number of windows across all subjects"""
        total = 0
        for sample in self.samples:
            n_win = sample['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            total += max(0, n_win)
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return single window: X=(n_ch, 5), y=label"""
        # Find which subject/window
        cumulative = 0
        sample = None
        rel_idx = 0
        
        for s in self.samples:
            n_win = s['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            
            if cumulative + n_win > idx:
                sample = s
                rel_idx = idx - cumulative + self.window_start
                break
            cumulative += n_win
        
        # SAFETY CHECK - prevent UnboundLocalError
        if sample is None or rel_idx >= sample['n_windows']:
            raise IndexError(f"Dataset index {idx} out of range (total len={len(self)})")
        
        # Load tensor
        tensor = torch.load(sample['path'])  # [n_windows, 19, 5]
        x = tensor[rel_idx, self.channels, :]  # [n_selected_ch, 5]
        
        # Label
        diagnosis = sample['diagnosis']
        label = 0 if diagnosis == 'AD' else 1
        
        return x.float(), torch.tensor(label, dtype=torch.long)
    
class STFTDataset(Dataset):
    """
    PyTorch Dataset for STFT spectrograms from preprocessed results.
    Shape: (n_windows, n_channels=19, n_freq=129, n_time=7) → [19,129,7] for 2D CNN
    """
    def __init__(
        self,
        results_root: str,
        subjects: Optional[List[int]] = None,
        diagnosis_filter: Optional[List[str]] = None,
        channels: Union[str, List[int]] = 'all',
        window_subset: Optional[Tuple[int, int]] = None,
    ):
        self.results_root = Path(results_root)
        self.data_dir = self.results_root / 'data'
        self.summary_path = self.results_root / 'tables' / 'stft_summary.csv'
        
        # Parse subjects + diagnoses
        self._build_samples(subjects, diagnosis_filter)
        
        # Channel selection
        self.channels = self._parse_channels(channels)
        self.n_channels = len(self.channels)
        
        # Window subset for testing
        self.window_start, self.window_end = window_subset or (0, None)
    
    def _build_samples(self, subjects, diagnosis_filter):
        """Scan .pt files → build (file_path, subject_id, diagnosis) list"""
        self.samples = []
        
        if self.summary_path.exists():
            summary_df = pd.read_csv(self.summary_path)
        else:
            summary_df = pd.DataFrame()
        
        # Find STFT files
        pt_files = sorted(self.data_dir.glob('sub-*_stft.pt'))
        print(f"Found {len(pt_files)} STFT files")
        
        for pt_file in pt_files:
            sub_str = pt_file.stem.split('_')[0]  # sub-001
            subject_id = int(sub_str.split('-')[1])
            
            if subjects is not None and subject_id not in subjects:
                continue
            
            diagnosis = get_diagnosis(subject_id)
            if diagnosis_filter is not None and diagnosis not in diagnosis_filter:
                print(f"Skipping {subject_id} ({diagnosis})")
                continue
            
            sub_row = summary_df[summary_df['subject_id'] == subject_id]
            if len(sub_row) > 0:
                n_windows = int(sub_row['n_windows'].iloc[0])
            else:
                tensor = torch.load(pt_file)
                n_windows = tensor.shape[0]
            
            self.samples.append({
                'path': pt_file,
                'subject_id': subject_id,
                'diagnosis': diagnosis,
                'n_windows': n_windows
            })
        
        print(f"✅ Loaded {len(self.samples)} subjects ({sum(s['n_windows'] for s in self.samples)} total windows)")
    
    def _parse_channels(self, channels):
        """Identical to other datasets"""
        if channels is None or (isinstance(channels, str) and channels.lower() == 'all'):
            return list(range(19))
        if isinstance(channels, (np.ndarray, list)):
            channels_list = [int(ch) for ch in list(channels)]
            if not all(0 <= ch < 19 for ch in channels_list):
                raise ValueError(f"Invalid channels {channels_list}: must be integers 0–18")
            return channels_list
        raise ValueError(f"Unknown channels type: {type(channels)} ({channels})")
    
    def __len__(self):
        total = 0
        for sample in self.samples:
            n_win = sample['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            total += max(0, n_win)
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return single spectrogram: X=[n_ch, 129, 7], y=label"""
        # Find subject/window (identical logic)
        cumulative = 0
        sample = None
        rel_idx = 0
        
        for s in self.samples:
            n_win = s['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            
            if cumulative + n_win > idx:
                sample = s
                rel_idx = idx - cumulative + self.window_start
                break
            cumulative += n_win
        
        if sample is None or rel_idx >= sample['n_windows']:
            raise IndexError(f"Dataset index {idx} out of range")
        
        # Load: [n_windows, 19, 129, 7] → [n_ch, 129, 7]
        tensor = torch.load(sample['path'])
        x = tensor[rel_idx, self.channels, :, :]  # [n_ch, 129freq, 7time]
        
        label = 0 if sample['diagnosis'] == 'AD' else 1
        return x.float(), torch.tensor(label, dtype=torch.long)


class ConnectivityDataset(Dataset):
    """
    PyTorch Dataset for Theta-PLV connectivity matrices.
    Shape: (n_windows, n_pairs=171) → flat vector for MLP/RF
    """
    def __init__(
        self,
        results_root: str,
        subjects: Optional[List[int]] = None,
        diagnosis_filter: Optional[List[str]] = None,
        window_subset: Optional[Tuple[int, int]] = None,
    ):
        self.results_root = Path(results_root)
        self.data_dir = self.results_root / 'data'
        self.summary_path = self.results_root / 'tables' / 'connectivity_summary.csv'
        
        # No channel selection needed (flat PLV pairs)
        self._build_samples(subjects, diagnosis_filter)
        self.window_start, self.window_end = window_subset or (0, None)
    
    def _build_samples(self, subjects, diagnosis_filter):
        """Scan .pt files → build (file_path, subject_id, diagnosis) list"""
        self.samples = []
        
        if self.summary_path.exists():
            summary_df = pd.read_csv(self.summary_path)
        else:
            summary_df = pd.DataFrame()
        
        # Find connectivity files
        pt_files = sorted(self.data_dir.glob('sub-*_connectivity.pt'))
        print(f"Found {len(pt_files)} connectivity files")
        
        for pt_file in pt_files:
            sub_str = pt_file.stem.split('_')[0]  # sub-001
            subject_id = int(sub_str.split('-')[1])
            
            if subjects is not None and subject_id not in subjects:
                continue
            
            diagnosis = get_diagnosis(subject_id)
            if diagnosis_filter is not None and diagnosis not in diagnosis_filter:
                print(f"Skipping {subject_id} ({diagnosis})")
                continue
            
            sub_row = summary_df[summary_df['subject_id'] == subject_id]
            if len(sub_row) > 0:
                n_windows = int(sub_row['n_windows'].iloc[0])
            else:
                tensor = torch.load(pt_file)
                n_windows = tensor.shape[0]
            
            self.samples.append({
                'path': pt_file,
                'subject_id': subject_id,
                'diagnosis': diagnosis,
                'n_windows': n_windows
            })
        
        print(f"✅ Loaded {len(self.samples)} subjects ({sum(s['n_windows'] for s in self.samples)} total windows)")
    
    def __len__(self):
        total = 0
        for sample in self.samples:
            n_win = sample['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            total += max(0, n_win)
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return single window: X=[171], y=label"""
        # Find subject/window (same logic as others)
        cumulative = 0
        sample = None
        rel_idx = 0
        
        for s in self.samples:
            n_win = s['n_windows']
            if self.window_end is not None:
                n_win = min(n_win - self.window_start, self.window_end - self.window_start)
            
            if cumulative + n_win > idx:
                sample = s
                rel_idx = idx - cumulative + self.window_start
                break
            cumulative += n_win
        
        if sample is None or rel_idx >= sample['n_windows']:
            raise IndexError(f"Dataset index {idx} out of range")
        
        # Load: [n_windows, 171] → [171]
        tensor = torch.load(sample['path'])
        x = tensor[rel_idx]  # [171]
        
        label = 0 if sample['diagnosis'] == 'AD' else 1
        return x.float(), torch.tensor(label, dtype=torch.long)
    
def raw_to_numpy(
    results_root: str,
    subjects: Optional[List[int]] = None,
    diagnosis_filter: Optional[List[str]] = None,
    channels: Union[str, List[int]] = 'all'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For sklearn RF baseline on raw windows: flatten [n_ch, 1000] → n_ch*1000 features
    WARNING: 19*1000=19K features/window = memory intensive!
    """
    dataset = RawDataset(results_root, subjects, diagnosis_filter, channels)
    
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x.view(-1).numpy())  # [n_ch, 1000] → [n_ch*1000]
        y_list.append(int(y))
    
    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"✅ Raw NumPy: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

def bandpower_to_numpy(
    results_root: str,
    subjects: Optional[List[int]] = None,
    diagnosis_filter: Optional[List[str]] = None,
    channels: Union[str, List[int]] = 'all'
    
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For sklearn models (RF/SVM): return X=(n_windows, n_features), y=(n_windows,)
    Flattens [n_ch, 5] → n_ch*5 features per window
    """
    dataset = BandpowerDataset(
        results_root, subjects, diagnosis_filter, channels
    )
    
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x.view(-1).numpy())  # Flatten [n_ch, 5] → [n_ch*5]
        y_list.append(int(y))
    
    X = np.stack(X_list)  # [n_windows, n_ch*5]
    y = np.array(y_list)
    print(f"✅ Numpy arrays: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

def connectivity_to_numpy(
    results_root: str,
    subjects: Optional[List[int]] = None,
    diagnosis_filter: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """For sklearn RF/SVM on connectivity: X=[n_windows, 171], y=[n_windows]"""
    dataset = ConnectivityDataset(results_root, subjects, diagnosis_filter)
    
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x.numpy())  # Already flat [171]
        y_list.append(int(y))
    
    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"✅ Connectivity NumPy: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

if __name__ == "__main__":
    conn_ds = ConnectivityDataset("results/connectivity_20260202_2157", diagnosis_filter=['AD', 'HC'])
    print(f"Connectivity length: {len(conn_ds)}")
    x, y = conn_ds[0]
    print(f"Sample shape: {x.shape}, label: {y}")
