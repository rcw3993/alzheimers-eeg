"""
Feature extraction: spectral bandpower via Welch's method.
Input:  windows_data dict from transforms.extract_raw_windows()
Output: (tensor [n_win, n_ch, n_bands], band_names)

Supports both absolute and relative bandpower:
  - Absolute: raw PSD power in each band (µV²/Hz)
  - Relative: each band divided by total broadband power (0-1, sums to 1)
              More robust to inter-subject amplitude scaling differences.
"""

import numpy as np
import torch
from scipy.signal import welch

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}


def compute_bandpower(
    windows_data: dict,
    sfreq: float,
    bands: dict = BANDS,
    relative: bool = False,
):
    """
    Compute bandpower for each window and channel.

    Args:
        windows_data:  dict from transforms.extract_raw_windows()
        sfreq:         sampling frequency in Hz
        bands:         dict of {name: (fmin, fmax)}
        relative:      if True, normalize each band by total broadband power
                       so values sum to 1 across bands per channel per window.
                       Removes inter-subject amplitude scaling differences.

    Returns:
        tensor:     FloatTensor (n_win, n_ch, n_bands)
        band_names: list[str]
    """
    win = windows_data["windows"]
    if isinstance(win, torch.Tensor):
        win = win.numpy()
    n_win, n_ch, n_samp = win.shape

    nperseg   = min(int(sfreq), n_samp)
    noverlap  = nperseg // 2
    band_names = list(bands.keys())
    n_bands   = len(band_names)
    bp = np.zeros((n_win, n_ch, n_bands), dtype=np.float32)

    for i in range(n_win):
        for ch in range(n_ch):
            freqs, psd = welch(
                win[i, ch], fs=sfreq, nperseg=nperseg,
                noverlap=noverlap, scaling="density"
            )
            df = freqs[1] - freqs[0]

            # Total broadband power (used for relative normalization)
            total_power = float(np.sum(psd) * df)

            for b_idx, (_, (fmin, fmax)) in enumerate(bands.items()):
                mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = float(np.sum(psd[mask]) * df)
                bp[i, ch, b_idx] = band_power

            if relative and total_power > 0:
                bp[i, ch, :] /= total_power

    return torch.from_numpy(bp), band_names