import pandas as pd
from pathlib import Path
import mne

# Global cache for participants data
_PARTICIPANTS_DF = None


def load_participants_tsv(data_root: str = "data") -> pd.DataFrame:
    """Load participants.tsv once and cache it."""
    global _PARTICIPANTS_DF
    if _PARTICIPANTS_DF is None:
        tsv_path = Path(data_root) / "ds004504" / "participants.tsv"
        if not tsv_path.exists():
            print(f"WARNING: {tsv_path} not found. Demographics will be unavailable.")
            _PARTICIPANTS_DF = pd.DataFrame()
        else:
            _PARTICIPANTS_DF = pd.read_csv(tsv_path, sep="\t")
            print(f"Loaded participants data: {len(_PARTICIPANTS_DF)} subjects")
    return _PARTICIPANTS_DF


def get_diagnosis(subject_id: int) -> str:
    """
    Derive diagnosis from subject ID using dataset convention.
    1–36  → AD
    37–65 → HC
    66–88 → FTD
    """
    if 1 <= subject_id <= 36:
        return "AD"
    elif 37 <= subject_id <= 65:
        return "HC"
    else:
        return "FTD"


def get_age(subject_id: int, data_root: str = "data"):
    """Get age from participants.tsv, returns None if unavailable."""
    df = load_participants_tsv(data_root)
    row = df[df["participant_id"] == f"sub-{subject_id:03d}"]
    return int(row.iloc[0]["Age"]) if len(row) > 0 else None


def get_gender(subject_id: int, data_root: str = "data"):
    """Get gender from participants.tsv, returns None if unavailable."""
    df = load_participants_tsv(data_root)
    row = df[df["participant_id"] == f"sub-{subject_id:03d}"]
    return row.iloc[0]["Gender"] if len(row) > 0 else None


def load_raw_eeg(subject_id: int, data_root: str = "data") -> dict | None:
    """
    Load EEG .set file for a subject and return a metadata dict.

    Returns None if the file is missing or fails to load.
    """
    eeg_path = (
        Path(data_root)
        / "ds004504"
        / f"sub-{subject_id:03d}"
        / "eeg"
        / f"sub-{subject_id:03d}_task-eyesclosed_eeg.set"
    )

    if not eeg_path.exists():
        print(f"ERROR: {eeg_path} not found")
        return None

    try:
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        print(f"Loaded subject {subject_id}: {raw.get_data().shape}, "
              f"{raw.times[-1] / 60:.1f} min")
    except Exception as e:
        print(f"ERROR loading subject {subject_id}: {e}")
        return None

    return {
        "raw": raw,
        "subject_id": subject_id,
        "diagnosis": get_diagnosis(subject_id),
        "demographics": {
            "age": get_age(subject_id, data_root),
            "gender": get_gender(subject_id, data_root),
        },
        "shape": raw.get_data().shape,
        "duration": raw.times[-1],
        "sfreq": raw.info["sfreq"],
    }


# ---------------------------------------------------------------------------
# ds006036 — same 88 subjects, eyes-open photic stimulation recording
# Files live under derivatives/eeglab/ with task-photomark naming
# ---------------------------------------------------------------------------

# Separate cache for ds006036 participants (same subjects, same TSV format)
_PARTICIPANTS_DF_006036 = None


def load_participants_tsv_006036(data_root: str = "data") -> pd.DataFrame:
    """Load participants.tsv for ds006036."""
    global _PARTICIPANTS_DF_006036
    if _PARTICIPANTS_DF_006036 is None:
        tsv_path = Path(data_root) / "ds006036" / "participants.tsv"
        if not tsv_path.exists():
            print(f"WARNING: {tsv_path} not found.")
            _PARTICIPANTS_DF_006036 = pd.DataFrame()
        else:
            _PARTICIPANTS_DF_006036 = pd.read_csv(tsv_path, sep="\t")
            print(f"Loaded ds006036 participants: {len(_PARTICIPANTS_DF_006036)} subjects")
    return _PARTICIPANTS_DF_006036


def load_raw_eeg_006036(subject_id: int, data_root: str = "data") -> dict | None:
    """
    Load eyes-open photic stimulation EEG from ds006036.

    Same 88 subjects as ds004504, same diagnosis mapping, same electrode
    layout. Files are under derivatives/eeglab/ with task-photomark naming.

    Use for zero-shot cross-condition generalization evaluation:
      - Train on ds004504 (eyes-closed)
      - Evaluate on ds006036 (eyes-open)
    """
    eeg_path = (
        Path(data_root)
        / "ds006036"
        / f"sub-{subject_id:03d}"
        / "eeg"
        / f"sub-{subject_id:03d}_task-photomark_eeg.set"
    )

    if not eeg_path.exists():
        print(f"ERROR: {eeg_path} not found")
        return None

    try:
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        print(f"Loaded ds006036 subject {subject_id}: {raw.get_data().shape}, "
              f"{raw.times[-1] / 60:.1f} min")
    except Exception as e:
        print(f"ERROR loading ds006036 subject {subject_id}: {e}")
        return None

    df = load_participants_tsv_006036(data_root)
    row = df[df["participant_id"] == f"sub-{subject_id:03d}"]

    return {
        "raw": raw,
        "subject_id": subject_id,
        "dataset": "ds006036",
        "diagnosis": get_diagnosis(subject_id),
        "demographics": {
            "age": int(row.iloc[0]["Age"]) if len(row) > 0 else None,
            "gender": row.iloc[0]["Gender"] if len(row) > 0 else None,
        },
        "shape": raw.get_data().shape,
        "duration": raw.times[-1],
        "sfreq": raw.info["sfreq"],
    }