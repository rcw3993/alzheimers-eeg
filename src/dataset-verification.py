import mne

raw = mne.io.read_raw_eeglab(
    "data/ds004504/sub-002/eeg/sub-002_task-eyesclosed_eeg.set",
    preload=True
)

print(raw)
