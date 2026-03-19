"""
Feature extraction: Short-Time Fourier Transform spectrograms.
Input:  windows_data dict from transforms.extract_raw_windows()
Output: (tensor [n_win, n_ch, n_freq, n_time], freqs, times)
"""

import numpy as np
import torch
from scipy.signal import stft as scipy_stft


def compute_stft(
    windows_data: dict,
    sfreq: float,
    nperseg: int = 256,
    noverlap: int = 128,
):
    """
    Compute log-power STFT spectrograms per window and channel.

    Returns:
        tensor: FloatTensor (n_win, n_ch, n_freq, n_time)
        freqs:  np.ndarray of frequency bins
        times:  np.ndarray of time bins
    """
    win = windows_data["windows"]
    if isinstance(win, torch.Tensor):
        win = win.numpy()
    n_win, n_ch, n_samp = win.shape

    # Infer output shape from first window
    freqs, times, Zxx = scipy_stft(
        win[0, 0], fs=sfreq, window="hann",
        nperseg=nperseg, noverlap=noverlap, boundary=None
    )
    n_freq, n_time = np.abs(Zxx).shape

    specs = np.zeros((n_win, n_ch, n_freq, n_time), dtype=np.float32)

    for i in range(n_win):
        for ch in range(n_ch):
            _, _, Zxx = scipy_stft(
                win[i, ch], fs=sfreq, window="hann",
                nperseg=nperseg, noverlap=noverlap, boundary=None
            )
            S = np.abs(Zxx) ** 2
            specs[i, ch] = np.log(S + 1e-10)

    return torch.from_numpy(specs), freqs, times