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


def apply_ica(raw, config: dict, subject_id) -> tuple:
    """
    Fit ICA on the filtered recording and remove artifact components
    identified by ICLabel.

    Must be called AFTER preprocess_raw() and BEFORE extract_raw_windows().
    ICA needs the full continuous recording to estimate components reliably —
    do not apply per-window.

    config keys used:
        ica:                        bool — whether to apply ICA at all
        ica_exclude:                list of component types to remove,
                                    e.g. ["eye", "muscle"]
                                    ICLabel labels: "brain", "eye", "muscle",
                                    "heart", "line_noise", "channel_noise", "other"
        ica_confidence_threshold:   float (0-1) — only remove a component if
                                    ICLabel confidence exceeds this value
                                    (default: 0.8)
        ica_n_components:           int or None — number of ICA components
                                    (default: None = use rank of data, typically 19)

    Returns:
        raw_clean:      MNE Raw object with artifact components removed
        ica_info:       dict with n_components_removed and which components
    """
    import mne
    try:
        from mne_icalabel import label_components
    except ImportError:
        raise ImportError(
            "mne-icalabel is required for ICA. Install with: pip install mne-icalabel"
        )

    exclude_types      = config.get("ica_exclude", ["eye", "muscle"])
    confidence_thresh  = float(config.get("ica_confidence_threshold", 0.8))
    n_components       = config.get("ica_n_components", None)

    print(f"  ICA: fitting on subject {subject_id} "
          f"(exclude={exclude_types}, confidence>={confidence_thresh})...")

    # --- Prepare a broadband copy that meets all ICLabel requirements ---
    # ICLabel needs:
    #   - 1-100 Hz bandpass (not 1-40 Hz)
    #   - CAR baked into the data array (not stored as projection)
    #   - Extended infomax decomposition
    #
    # We can't recover 40-100 Hz from the already-filtered `raw`, so we
    # reload from the original file and apply a fresh broadband filter.
    # ICA component labels are then applied to the original `raw`.
    eeg_path = raw.filenames[0] if hasattr(raw, "filenames") and raw.filenames else None

    if eeg_path is not None:
        raw_for_ica = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        raw_for_ica.filter(l_freq=1, h_freq=100, verbose=False)
        # Use projection=False to bake CAR directly into data array.
        # ICLabel checks raw.info['custom_ref_applied'] which only gets set
        # when the reference is applied directly, not via SSP projection.
        raw_for_ica.set_eeg_reference('average', projection=False, verbose=False)
    else:
        # Fallback: no source file available (e.g. passed in programmatically)
        # Use the already-filtered raw — warnings will appear but ICA still runs
        print(f"    WARNING: Cannot reload raw from file for broadband ICA fit. "
              f"ICLabel accuracy may be reduced.")
        raw_for_ica = raw.copy()
        raw_for_ica.apply_proj()

    # --- Fit ICA using extended infomax (what ICLabel was trained on) ---
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw_for_ica, verbose=False)

    # ICLabel: classify components using the broadband copy
    component_labels = label_components(raw_for_ica, ica, method="iclabel")
    labels      = component_labels["labels"]       # list of str, one per component
    probs       = component_labels["y_pred_proba"] # array (n_components, n_classes)

    # Print full ICLabel breakdown so we can see what it found
    # Note: mne-icalabel returns keys 'labels' and 'y_pred_proba' only.
    # The predicted label confidence is probs[i].max() since the label
    # IS the argmax class.
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    top_conf = [(labels[i], float(probs[i].max())) for i in range(len(labels))]
    top_conf.sort(key=lambda x: -x[1])
    print(f"    ICLabel summary: {label_counts}")
    print(f"    Top-5 by confidence: {top_conf[:5]}")
    print(f"    Available keys: {list(component_labels.keys())}")

    # Find components to exclude: type matches AND confidence exceeds threshold
    # The confidence for a component is simply the max probability value,
    # since labels[i] = argmax(probs[i]).
    exclude_idx = []
    for idx, label in enumerate(labels):
        if label in exclude_types:
            confidence = float(probs[idx].max())
            if confidence >= confidence_thresh:
                exclude_idx.append(idx)
                print(f"    Excluding component {idx}: {label} "
                      f"(confidence={confidence:.2f})")

    if not exclude_idx:
        print(f"    No components exceeded confidence threshold — signal unchanged")

    ica.exclude = exclude_idx
    # Apply removal to the original 1-40 Hz filtered data (not the broadband copy)
    raw_clean = ica.apply(raw.copy(), verbose=False)

    ica_info = {
        "n_components_total":   ica.n_components_,
        "n_components_removed": len(exclude_idx),
        "excluded_indices":     exclude_idx,
        "excluded_labels":      [labels[i] for i in exclude_idx],
    }

    print(f"  ICA: removed {len(exclude_idx)}/{ica.n_components_} components")
    return raw_clean, ica_info


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