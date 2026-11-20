# ML pipeline: healthy movement signature + PD duration context

The pipeline has two stages:

1. **Stage 1, healthy signature:** learn what "normal" movement looks like using **healthy participants only**.
2. **Stage 2, PD scoring and duration context:** score PD patients against the healthy reference, then compare each patient to other patients with a similar disease duration.

The same feature extraction and scoring functions are used for healthy and PD participants.

## Folder contents

```
ml_pipeline/
├── healthy_signature_v2.py          # pipeline code (also required to load the model)
├── models/
│   ├── healthy_model_v2.joblib      # frozen Stage 1 model
│   └── stage2_duration_context.joblib
└── outputs/
    ├── healthy/
    │   ├── healthy_unit_features_v2.csv
    │   ├── healthy_oof_scores_v2.csv
    │   └── healthy_subject_deviation_v2.csv
    ├── pd/
    │   ├── pd_unit_scores.csv
    │   └── pd_scores.csv
    └── figures/
        └── pd_context_figure.png
```

## How Stage 1 works (`healthy_signature_v2.py`)

1. **Prepare segments.** Keep the first 974 samples (dropping 50 padding samples). For CrossArms and LiftHold, also trim the first 3s, because movement starts later in those tasks.
2. **Extract features.** There are 22 features per segment, 11 each for the accelerometer and gyroscope: log RMS, log 5–95% range, log jerk, log band power (0.5–3, 3–7, 4–6, 7–12, 12–20 Hz), relative power in 4–6 and 3–7 Hz, and spectral entropy (Welch PSD, 0.5–20 Hz).
3. **Build the unit table.** Average the features for each **unit**, meaning one (subject, task, wrist role). That gives 11 tasks × 2 roles = 22 units per person. Age and sex are attached to each row.
4. **Fit one normative model per (task, role)**, each on healthy participants only:
   - median imputation
   - ridge regression (α = 10) to remove age and sex effects
   - robust scaling, then clipping at ±8
   - Gaussian model with Ledoit–Wolf shrinkage covariance
   - raw score = log Mahalanobis distance
5. **Out-of-fold calibration.** Run 5-fold cross-validation split by subject, repeated 3 times. This means each healthy person is scored by models that never saw them. Raw scores are converted to z-scores using the median and MAD of those out-of-fold scores, then clipped to [-3, 10].
6. **Freeze.** Save all 22 unit models, the calibration table, the config and the package versions to `healthy_model_v2.joblib`.

A subject's **deviation** is the mean z-score across all of that subject's units.

## Stage 2

- `stage2_duration_context.joblib` contains quantile regression curves of PD deviation vs. log(1 + years since diagnosis) (`curves`, `levels`) and healthy reference scores (`healthy_ref`).
- `pd_scores.csv` gives each patient's deviation, their percentile vs. healthy controls, and an out-of-fold peer percentile among patients with the same disease duration.

<!-- TODO: the code that builds stage2_duration_context.joblib, pd_scores.csv and pd_context_figure.png is not in this repo yet. Add that script here. -->

## Output files

| File | One row = | Key columns |
|---|---|---|
| `healthy_unit_features_v2.csv` | healthy subject × task × role | the 22 features, `age`, `sex_male` |
| `healthy_oof_scores_v2.csv` | healthy subject × task × role | `raw` (log Mahalanobis), `mu`/`sd` (calibration), `z` |
| `healthy_subject_deviation_v2.csv` | healthy subject (79) | `deviation` (mean z), `n_units`, `age`, `gender` |
| `pd_unit_scores.csv` | PD subject × task × role | `raw`, `mu`, `sd`, `z` |
| `pd_scores.csv` | PD subject (269) | `deviation`, `duration` (years), `percentile_vs_healthy`, `peer_percentile_same_duration` |

## How to run

Install the pinned packages first, from the repo root:

```bash
pip install -r requirements.txt
```

> The model must be loaded with **scikit-learn 1.6.1**. Newer versions fail with an error like `'SimpleImputer' object has no attribute '_fill_dtype'`.

**Score participants with the frozen model** (run from the repo root):

```python
import sys; sys.path.append("ml_pipeline")   # the model needs this module to load
import joblib
import healthy_signature_v2 as hs

X, meta = hs.load_data("data/preprocessed/segments.npy",
                       "data/preprocessed/segment_metadata.csv")
bundle = joblib.load("ml_pipeline/models/healthy_model_v2.joblib")

pd_mask = (meta["label"] == 1).to_numpy()
unit_scores, subject_scores = hs.score_participants(
    bundle, X[pd_mask], meta[pd_mask].reset_index(drop=True))
```

**Rebuild the healthy model from scratch:**

```python
bundle, unit, oof, subj = hs.build_healthy_model(
    X, meta, out_path="ml_pipeline/models/healthy_model_v2.joblib")
```
