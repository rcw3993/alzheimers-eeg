"""
Feature extraction: spectral bandpower via Welch's method.
Input:  windows_data dict from transforms.extract_raw_windows()
Output: (tensor [n_win, n_ch, n_bands], band_names)
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
 
 
def compute_bandpower(windows_data: dict, sfreq: float, bands: dict = BANDS):
    """
    Compute absolute bandpower for each window and channel.
 
    Returns:
        tensor:     FloatTensor (n_win, n_ch, n_bands)
        band_names: list[str]
    """
    win = windows_data["windows"]
    if isinstance(win, torch.Tensor):
        win = win.numpy()
    n_win, n_ch, n_samp = win.shape
 
    nperseg = min(int(sfreq), n_samp)
    noverlap = nperseg // 2
    band_names = list(bands.keys())
    bp = np.zeros((n_win, n_ch, len(band_names)), dtype=np.float32)
 
    for i in range(n_win):
        for ch in range(n_ch):
            freqs, psd = welch(
                win[i, ch], fs=sfreq, nperseg=nperseg,
                noverlap=noverlap, scaling="density"
            )
            df = freqs[1] - freqs[0]
            for b_idx, (_, (fmin, fmax)) in enumerate(bands.items()):
                mask = (freqs >= fmin) & (freqs <= fmax)
                bp[i, ch, b_idx] = np.sum(psd[mask]) * df
 
    return torch.from_numpy(bp), band_names
 