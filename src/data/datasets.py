"""
PyTorch Dataset classes for each EEG representation.

All datasets share a common base that handles:
  - scanning .pt files in an outputs/ feature directory
  - subject / diagnosis filtering
  - window subsetting
  - __len__ / __getitem__ indexing logic

Adding a new representation = subclass BaseEEGDataset and implement
_pt_suffix() and _load_window().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union

from src.data.loaders import get_diagnosis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_channels(channels, n_total: int = 19) -> List[int]:
    """Normalize channel argument to a plain list of ints."""
    if channels is None or (isinstance(channels, str) and channels.lower() == "all"):
        return list(range(n_total))
    if isinstance(channels, (np.ndarray, list)):
        ch_list = [int(c) for c in channels]
        if not all(0 <= c < n_total for c in ch_list):
            raise ValueError(f"Channel indices must be in 0–{n_total - 1}, got {ch_list}")
        return ch_list
    raise TypeError(f"Unsupported channels type: {type(channels)}")


def _label(diagnosis: str) -> int:
    """Binary label: AD=0, HC=1."""
    return 0 if diagnosis == "AD" else 1


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEEGDataset(Dataset):
    """
    Shared scaffolding for all EEG representation datasets.

    Subclasses must implement:
        _pt_suffix()   → str  e.g. "_bandpower.pt"
        _load_window() → torch.Tensor  given (tensor, rel_idx, channels)
    """

    def __init__(
        self,
        features_dir: str,
        subjects: Optional[List[int]] = None,
        diagnosis_filter: Optional[List[str]] = None,
        channels: Union[str, List[int]] = "all",
        window_subset: Optional[Tuple[int, int]] = None,
    ):
        self.features_dir = Path(features_dir) / "data"
        self.channels = _parse_channels(channels)
        self.window_start, self.window_end = window_subset or (0, None)
        self._build_samples(subjects, diagnosis_filter)

    # -- subclass interface --------------------------------------------------

    def _pt_suffix(self) -> str:
        raise NotImplementedError

    def _load_window(self, tensor: torch.Tensor, rel_idx: int) -> torch.Tensor:
        raise NotImplementedError

    # -- internal ------------------------------------------------------------

    def _build_samples(self, subjects, diagnosis_filter):
        self.samples = []
        pt_files = sorted(self.features_dir.glob(f"sub-*{self._pt_suffix()}"))
        if not pt_files:
            raise FileNotFoundError(
                f"No files matching 'sub-*{self._pt_suffix()}' in {self.features_dir}"
            )

        for pt_file in pt_files:
            subject_id = int(pt_file.stem.split("_")[0].split("-")[1])

            if subjects is not None and subject_id not in subjects:
                continue

            diagnosis = get_diagnosis(subject_id)
            if diagnosis_filter is not None and diagnosis not in diagnosis_filter:
                continue

            # Prefer reading n_windows from the summary CSV to avoid loading all tensors
            n_windows = self._get_n_windows(pt_file, subject_id)
            self.samples.append(
                dict(path=pt_file, subject_id=subject_id, diagnosis=diagnosis, n_windows=n_windows)
            )

        total_win = sum(s["n_windows"] for s in self.samples)
        print(f"{self.__class__.__name__}: {len(self.samples)} subjects, {total_win} windows")

    def _get_n_windows(self, pt_file: Path, subject_id: int) -> int:
        """Read n_windows from summary CSV if available, else load the tensor."""
        summary_csv = pt_file.parent.parent / "tables" / "summary.csv"
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            row = df[df["subject_id"] == subject_id]
            if len(row) > 0:
                return int(row.iloc[0]["n_windows"])
        tensor = torch.load(pt_file, weights_only=True)
        return tensor.shape[0]

    def _effective_n_windows(self, sample: dict) -> int:
        n = sample["n_windows"] - self.window_start
        if self.window_end is not None:
            n = min(n, self.window_end - self.window_start)
        return max(0, n)

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return sum(self._effective_n_windows(s) for s in self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cumulative = 0
        for sample in self.samples:
            n = self._effective_n_windows(sample)
            if cumulative + n > idx:
                rel_idx = idx - cumulative + self.window_start
                tensor = torch.load(sample["path"], weights_only=True)
                x = self._load_window(tensor, rel_idx)
                y = torch.tensor(_label(sample["diagnosis"]), dtype=torch.long)
                return x.float(), y
            cumulative += n
        raise IndexError(f"Index {idx} out of range (dataset length {len(self)})")


# ---------------------------------------------------------------------------
# Concrete dataset classes
# ---------------------------------------------------------------------------

class RawDataset(BaseEEGDataset):
    """Raw windows: (n_windows, n_channels, n_timepoints=1000)."""

    def _pt_suffix(self): return "_raw.pt"

    def _load_window(self, tensor, rel_idx):
        return tensor[rel_idx][self.channels]  # [n_ch, 1000]


class BandpowerDataset(BaseEEGDataset):
    """Bandpower features: (n_windows, n_channels, n_bands=5)."""

    def _pt_suffix(self): return "_bandpower.pt"

    def _load_window(self, tensor, rel_idx):
        return tensor[rel_idx][self.channels]  # [n_ch, 5]


class STFTDataset(BaseEEGDataset):
    """STFT spectrograms: (n_windows, n_channels, n_freq=129, n_time=7)."""

    def _pt_suffix(self): return "_stft.pt"

    def _load_window(self, tensor, rel_idx):
        return tensor[rel_idx][self.channels]  # [n_ch, 129, 7]


class ConnectivityDataset(BaseEEGDataset):
    """Theta-PLV pairs: (n_windows, n_pairs=171). No channel selection."""

    def _pt_suffix(self): return "_connectivity.pt"

    def _load_window(self, tensor, rel_idx):
        return tensor[rel_idx]  # [171]


# ---------------------------------------------------------------------------
# NumPy helpers for sklearn models
# ---------------------------------------------------------------------------

def _dataset_to_numpy(
    dataset: BaseEEGDataset, flatten: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x.reshape(-1).numpy() if flatten else x.numpy())
        y_list.append(int(y))
    return np.stack(X_list), np.array(y_list)


def bandpower_to_numpy(
    features_dir: str,
    subjects: Optional[List[int]] = None,
    diagnosis_filter: Optional[List[str]] = None,
    channels: Union[str, List[int]] = "all",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X=(n_windows, n_ch*5), y=(n_windows,) for sklearn."""
    ds = BandpowerDataset(features_dir, subjects, diagnosis_filter, channels)
    X, y = _dataset_to_numpy(ds, flatten=True)
    print(f"bandpower_to_numpy: X={X.shape}, y={y.shape}")
    return X, y


def connectivity_to_numpy(
    features_dir: str,
    subjects: Optional[List[int]] = None,
    diagnosis_filter: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X=(n_windows, 171), y=(n_windows,) for sklearn."""
    ds = ConnectivityDataset(features_dir, subjects, diagnosis_filter)
    X, y = _dataset_to_numpy(ds, flatten=False)
    print(f"connectivity_to_numpy: X={X.shape}, y={y.shape}")
    return X, y


def raw_to_numpy(
    features_dir: str,
    subjects: Optional[List[int]] = None,
    diagnosis_filter: Optional[List[str]] = None,
    channels: Union[str, List[int]] = "all",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X=(n_windows, n_ch*1000), y=(n_windows,). Memory-intensive."""
    ds = RawDataset(features_dir, subjects, diagnosis_filter, channels)
    X, y = _dataset_to_numpy(ds, flatten=True)
    print(f"raw_to_numpy: X={X.shape}, y={y.shape}")
    return X, y