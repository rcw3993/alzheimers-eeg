import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import mne


def preprocess_raw(subject_data: dict, config: dict):
    """
    Apply bandpass filter and average reference from config.

    config keys used:
        filters: [l_freq, h_freq]  e.g. [1, 40]

    Returns a filtered/referenced MNE Raw object.
    """
    raw = subject_data["raw"].copy()

    l_freq, h_freq = config["filters"]
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj()

    return raw


def extract_raw_windows(raw, config: dict, subject_id: int) -> dict:
    """
    Slide a window across the recording and return a dict of tensors.

    config keys used:
        window_size:        window length in seconds (e.g. 2)
        step_size:          step in seconds (e.g. 0.5)
        artifact_rejection: bool — skip windows with peak-to-peak > ptp_threshold
        ptp_threshold:      peak-to-peak threshold in volts (e.g. 100e-6)

    Returns:
        windows:       FloatTensor (n_windows, n_channels, window_samples)
        window_times:  np.ndarray of window start times in seconds
        window_samples: int
        sfreq:         float
    """
    sfreq = raw.info["sfreq"]
    window_samples = int(config["window_size"] * sfreq)
    step_samples = int(config.get("step_size", 0.5) * sfreq)
    reject_artifacts = config.get("artifact_rejection", False)
    ptp_thresh = float(config.get("ptp_threshold", 100e-6))

    data = raw.get_data()  # (n_channels, n_times)
    windows, window_times = [], []
    total_possible = 0

    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        total_possible += 1
        window = data[:, start : start + window_samples]

        if reject_artifacts:
            ptp = window.max(axis=1) - window.min(axis=1)
            if (ptp > ptp_thresh).any():
                continue

        # Z-score per channel per window
        window = (window - window.mean(axis=1, keepdims=True)) / (
            window.std(axis=1, keepdims=True) + 1e-8
        )
        windows.append(window)
        window_times.append(start / sfreq)

    windows_arr = np.stack(windows)
    windows_tensor = torch.FloatTensor(windows_arr)

    n_rejected = total_possible - len(windows)
    print(f"Subject {subject_id}: {len(windows)} windows "
          f"({n_rejected} rejected)" if reject_artifacts else
          f"Subject {subject_id}: {len(windows)} windows")

    return {
        "windows": windows_tensor,       # (n_windows, n_channels, window_samples)
        "window_times": np.array(window_times),
        "window_samples": window_samples,
        "sfreq": sfreq,
    }


def save_sample_plots(windows_data: dict, subject_data: dict, output_dir: str):
    """Save a figure showing 3 representative windows for 3 channels."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_channels = [0, 9, 18]  # Fp1, Cz, O2

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i, ch_idx in enumerate(sample_channels):
        ax = axes[i]
        for w_idx in range(min(3, len(windows_data["windows"]))):
            window = windows_data["windows"][w_idx, ch_idx].numpy()
            t = np.linspace(0, windows_data["window_samples"] / windows_data["sfreq"], len(window))
            ax.plot(t, window + w_idx * 3, linewidth=0.8)
        ch_name = subject_data["raw"].ch_names[ch_idx]
        ax.set_title(f"{ch_name} ({subject_data['diagnosis']})")
        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plot_path = Path(output_dir) / f"subject_{subject_data['subject_id']}_windows.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")