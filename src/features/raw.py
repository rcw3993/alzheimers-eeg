"""
'Feature extraction' for raw windows — just a passthrough.
Included for consistency so all representations share the same interface.
"""

import torch


def compute_raw(windows_data: dict, **kwargs):
    """
    Return raw windows unchanged.

    Returns:
        tensor: FloatTensor (n_win, n_ch, n_timepoints)
    """
    win = windows_data["windows"]
    if not isinstance(win, torch.Tensor):
        win = torch.FloatTensor(win)
    return win