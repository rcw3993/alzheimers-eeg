# EEG Alzheimer's Detection

A reproducible machine learning pipeline for early Alzheimer's disease detection
using EEG signals. Compares multiple signal representations and model architectures
under rigorous subject-wise cross-validation, then evaluates cross-condition
generalization on an independent recording paradigm.

---

## Results

### Primary model: Bandpower + Random Forest (5 electrodes)

| Metric | Value |
|---|---|
| Cross-validation AUC | 0.815 ± 0.036 |
| Sensitivity (screening-optimized) | 89.7% |
| Specificity | 62.2% |
| Decision threshold | 0.362 (Youden's J, sensitivity ≥ 0.85) |
| Electrodes used | 5 (T4, Pz, Cz, T6, C3) |

### Representation comparison (subject-wise 5-fold CV, 65 subjects)

| Representation | Model | AUC | Electrodes |
|---|---|---|---|
| Bandpower | Random Forest | 0.815 ± 0.04 | 5 |
| Connectivity (PLV) | Random Forest | 0.769 ± 0.05 | 19 |
| STFT spectrograms | 2D CNN | 0.747 ± 0.06 | 19 |
| Raw windows | 1D CNN | 0.508 ± 0.07 | 19 |

### Preprocessing experiments

| Pipeline | AUC | Sensitivity | Notes |
|---|---|---|---|
| Baseline (no ICA) | 0.815 | 89.7% | 89,239 windows |
| + ICA artifact removal | 0.810 | 91.8% | 97,846 windows (+9.6%) |
| + Relative bandpower | 0.814 | 91.5% | Same window count as ICA |

### Electrode reduction sweep

Reducing from 19 electrodes to 5 (temporal, parietal, central regions) slightly
**improved** cross-validation AUC (0.801 → 0.822), suggesting redundant or noisy
electrodes hurt generalization. STFT + 2D CNN showed the same pattern, peaking at
k=3 electrodes (AUC 0.791), supporting electrode reduction as representation-agnostic.

### Cross-condition generalization (zero-shot)

Trained on ds004504 (eyes-closed resting-state), evaluated without fine-tuning on
ds006036 (eyes-open photic stimulation), same 65 subjects:

| Metric | Value |
|---|---|
| AUC | 0.979 |
| Accuracy | 89.2% |
| Sensitivity | 94.4% |
| Specificity | 82.8% |
| Correct predictions | 58 / 65 |

The AD bandpower signature is stable across recording paradigms for the same
individuals. Note: this is cross-condition generalization, not cross-site — both
datasets share the same cohort.

### Feature importance (post-ICA)

ICA removed 1–12 components per subject (primarily eye blink and line noise),
increasing usable windows by ~9.6%. Post-ICA feature importance shifted from
beta-band temporal activity (T4 beta dominant pre-ICA) to alpha-band parietal
activity (Pz alpha dominant post-ICA). Posterior alpha slowing is the most
consistently reported EEG biomarker in Alzheimer's literature, suggesting ICA
produced a more biologically valid feature representation.

### Negative results (informative)

- **Relative bandpower normalization** did not reduce fold-to-fold AUC variance
  (range 0.714–0.924), suggesting the variance is biological (genuine heterogeneity
  in AD progression) rather than technical (inter-subject amplitude scaling).
- **ICA** did not significantly improve AUC despite removing real artifacts, because
  peak-to-peak artifact rejection was already handling most contaminated windows.
- **Raw 1D CNN** (AUC 0.508) confirms deep learning requires more subjects than n=65
  to outperform classical spectral features on this task.

---

## Motivation

Alzheimer's disease affects over 55 million people worldwide. Gold-standard screening
tools (PET imaging, CSF analysis) are accurate but expensive, invasive, and scarce.
EEG is inexpensive, non-invasive, and widely available — making it a realistic
candidate for early screening if reliable computational biomarkers can be established.
This project investigates whether a 5-electrode EEG system with bandpower features
can serve as a practical early screening tool, and what validation steps would be
required before clinical deployment.

---

## Limitations

This project is a research prototype, not a clinical tool. Key limitations:

- **Single cohort**: both ds004504 and ds006036 contain the same 88 subjects.
  The zero-shot result (AUC 0.979) reflects cross-paradigm generalization within
  a cohort, not cross-site generalization across independent populations.
- **Sample size**: n=65 (AD + HC) is modest. Fold-to-fold AUC variance (0.714–0.924)
  indicates results are sensitive to which subjects fall in which fold.
- **Resting-state only**: all recordings are eyes-closed or eyes-open resting-state.
  No task-based or longitudinal data.
- **No clinical validation**: no IRB oversight, no clinician review, no comparison
  against established screening instruments (MMSE, MoCA).
- **Dataset acquisition protocol**: both datasets were recorded at a single site
  under controlled conditions. Real-world EEG recordings vary substantially in
  quality, electrode placement, and equipment.

**Proposed next steps** to address these limitations: multi-center validation on
independently collected datasets, domain adaptation for cross-site generalization,
healthy life years (QALY) economic modeling, and decision-theoretic threshold
selection under resource constraints.

---

## Data

This project uses two OpenNeuro datasets from the same 88-subject cohort:

**ds004504** — Eyes-closed resting-state EEG (training dataset):
```bash
openneuro download ds004504 data/ds004504
```
Or: `https://openneuro.org/datasets/ds004504`

**ds006036** — Eyes-open photic stimulation EEG (zero-shot evaluation):
```bash
# Install openneuro-py first: pip install openneuro-py
python -c "import openneuro; openneuro.download(dataset='ds006036', target_dir='data/ds006036')"
```
Or: `https://openneuro.org/datasets/ds006036`

Both datasets are not included in this repository. Place them under `data/` so paths
match what the code expects. Subjects 1–36 are AD, 37–65 are HC, 66–88 are FTD
(FTD subjects are excluded from binary classification experiments).

**Expected structure:**
```
data/
├── ds004504/
│   ├── participants.tsv
│   └── sub-001/eeg/sub-001_task-eyesclosed_eeg.set
│   └── ...
└── ds006036/
    ├── participants.tsv
    └── sub-001/eeg/sub-001_task-photomark_eeg.set
    └── ...
```

---

## Setup

```bash
git clone <repo-url>
cd eeg-alzheimers

# Option A: pip
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate eeg-alzheimers
```

---

## Usage

### Full experiment (one command)

```bash
python scripts/run_experiment.py configs/experiments/bandpower_rf_v1.yaml
```

### Step by step

```bash
# 1. Extract features (test on 3 subjects first)
python scripts/extract_features.py configs/features/bandpower.yaml --subjects 1 2 3
python scripts/extract_features.py configs/features/bandpower.yaml

# 2. Train and evaluate
python scripts/train.py configs/training/bandpower_rf_thresh_v1.yaml --save-model

# 3. Generate result figures
python scripts/evaluate.py configs/evaluation/bandpower_rf_thresh_v1.yaml

# 4. Zero-shot cross-condition evaluation
python scripts/evaluate_zeroshot.py

# 5. Inference on a single recording
python scripts/predict.py --subject 1
python scripts/predict.py --set-file path/to/recording.set
```

### Available feature configs

| Config | Description |
|---|---|
| `configs/features/bandpower.yaml` | Absolute bandpower, no ICA |
| `configs/features/bandpower_ica.yaml` | Absolute bandpower + ICA artifact removal |
| `configs/features/bandpower_rel.yaml` | Relative bandpower + ICA |
| `configs/features/stft.yaml` | STFT spectrograms |
| `configs/features/connectivity.yaml` | Theta-band PLV connectivity |
| `configs/features/raw.yaml` | Raw windowed EEG |

---

## Project structure

```
eeg-alzheimers/
│
├── data/                          # gitignored — download separately
│   ├── ds004504/                  # training dataset
│   └── ds006036/                  # zero-shot evaluation dataset
│
├── outputs/                       # gitignored — generated by pipeline
│   ├── features/                  # extracted .pt tensors per subject
│   ├── models/                    # saved .joblib model weights
│   └── results/                   # CV metrics, figures, experiment logs
│
├── configs/
│   ├── features/                  # one YAML per representation/preprocessing variant
│   ├── training/                  # one YAML per model configuration
│   ├── evaluation/                # one YAML per evaluation run
│   └── experiments/               # end-to-end experiment YAMLs
│
├── src/
│   ├── data/
│   │   ├── loaders.py             # load .set files from ds004504 and ds006036
│   │   ├── transforms.py          # bandpass filter, ICA, windowing
│   │   └── datasets.py            # PyTorch Dataset classes (BaseEEGDataset)
│   ├── features/
│   │   ├── bandpower.py           # Welch PSD, absolute and relative
│   │   ├── stft.py                # log-power STFT spectrograms
│   │   ├── connectivity.py        # theta-band PLV
│   │   └── raw.py                 # passthrough
│   ├── models/
│   │   ├── rf.py                  # RandomForestWrapper with channel importances
│   │   ├── cnn1d.py               # 1D CNN for raw windows
│   │   ├── cnn2d.py               # 2D CNN for STFT spectrograms
│   │   └── mlp.py                 # MLP for PLV features
│   ├── training/
│   │   ├── cross_val.py           # subject-wise GroupKFold CV
│   │   └── threshold.py           # sensitivity-first threshold optimization
│   ├── evaluation/
│   │   ├── importance.py          # RF feature importance, electrode sweep
│   │   └── plots.py               # shared plotting functions
│   └── utils.py                   # set_seed, compute_metrics, log_experiment
│
├── scripts/
│   ├── extract_features.py        # Step 1: raw EEG → .pt feature tensors
│   ├── train.py                   # Step 2: features → CV results + saved model
│   ├── evaluate.py                # Step 3: saved model → result figures
│   ├── evaluate_zeroshot.py       # Zero-shot evaluation on ds006036
│   ├── evaluate_stft.py           # STFT electrode sweep
│   ├── predict.py                 # Inference on a single .set file
│   └── run_experiment.py          # Steps 1+2 combined from a single config
│
├── environment.yml                # conda environment
├── requirements.txt               # pip dependencies (unpinned)
├── requirements_pinned.txt        # pip dependencies (exact versions)
└── README.md
```

---

## Adding a new representation

1. Add `compute_X(windows_data, sfreq, **kwargs)` in `src/features/X.py`
2. Register it in `scripts/extract_features.py` → `REPRESENTATION_FN`
3. Subclass `BaseEEGDataset` in `src/data/datasets.py` (implement `_pt_suffix` and `_load_window`)
4. Add `configs/features/X.yaml`

## Adding a new dataset

1. Drop the dataset folder under `data/`
2. Add a `load_raw_eeg_DATASETID()` function in `src/data/loaders.py`
3. Add or extend `get_diagnosis()` for the new subject ID convention
4. Create feature configs pointing at the new subjects

---

## Reproducibility

All experiments are seeded (`set_seed(42)`). Subject-wise `GroupKFold` ensures no
subject appears in both train and test splits. Configs are the single source of truth
for all hyperparameters — no hardcoded values in source files. Every training run
appends a row to `outputs/results/<run>/experiment_logs.csv`.

## Hardware and runtime (approximate)

Tested on: AMD Ryzen 9 9950X3D, RTX 5090, 64GB RAM, Windows 11

| Step | Runtime |
|---|---|
| Feature extraction, 65 subjects, bandpower (no ICA) | ~3 min |
| Feature extraction, 65 subjects, bandpower + ICA | ~45 min |
| Training, bandpower RF, 65 subjects | ~2 min |
| Electrode sweep (6 k-values × 5 folds) | ~15 min |
| Zero-shot evaluation, 65 subjects | ~8 min |

Random Forest training is CPU-bound (`n_jobs=-1` uses all cores). GPU is used for
CNN training only. ICA fitting (extended infomax) is CPU-bound regardless of GPU.
