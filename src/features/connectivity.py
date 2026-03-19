"""
Feature extraction: Theta-band Phase Locking Value (PLV) connectivity.
Input:  windows_data dict from transforms.extract_raw_windows()
Output: (tensor [n_win, n_pairs], pairs list)
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert


def _bandpass(x: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4):
    nyq = 0.5 * sfreq
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, x)


def compute_plv(
    windows_data: dict,
    sfreq: float,
    fmin: float = 4.0,
    fmax: float = 8.0,
):
    """
    Compute pairwise theta-band PLV for each window.

    Returns:
        tensor: FloatTensor (n_win, n_pairs)  where n_pairs = n_ch*(n_ch-1)/2
        pairs:  list of (i, j) channel index tuples
    """
    win = windows_data["windows"]
    if isinstance(win, torch.Tensor):
        win = win.numpy()
    n_win, n_ch, n_samp = win.shape

    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    plv_feats = np.zeros((n_win, len(pairs)), dtype=np.float32)

    for w in range(n_win):
        filtered = np.stack([
            _bandpass(win[w, ch], sfreq, fmin, fmax)
            for ch in range(n_ch)
        ])
        phases = np.angle(hilbert(filtered, axis=1))

        for p, (i, j) in enumerate(pairs):
            dphi = phases[i] - phases[j]
            plv_feats[w, p] = float(np.abs(np.exp(1j * dphi).mean()))

    return torch.from_numpy(plv_feats), pairs