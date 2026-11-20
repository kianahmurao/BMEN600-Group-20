# Data

All files are in `preprocessed/`. These are the inputs to the ML pipeline.

| File | What it is | Where to find |
|---|---|---|
| `segments.npy` | Array containing one row per movement segment: 1024 samples at 100 Hz × 6 channels (acc x/y/z, gyro x/y/z) | **[Download here](https://drive.google.com/file/d/1XAH6HQQ__LVGcHmAVmhQo8CnJjl5f8YQ/view?usp=sharing)** and place it in `data/preprocessed/` |
| `segment_metadata.csv` | One row per movement segment (row *i* describes `segments[i]`): subject, task, wrist, dominant/non-dominant role, label, age, diagnosis age, disease duration, gender, handedness, source file. | Within this repo |
| `qc_report.csv` | Quality-control results for each raw recording file: lengths, missing-value fraction, per-channel SD, max acceleration/rotation, padding, pass/fail and reason. 10,318 files, 10,317 passed (1 failed for accelerometer saturation). | Within this repo |

**Labels:** `0` = healthy control (79 subjects), `1` = Parkinson's disease (276 subjects), `2` = other diagnosis (114 subjects). 

**Tasks (11):** CrossArms, DrinkGlas, Entrainment, HoldWeight, LiftHold, PointFinger, Relaxed, RelaxedTask, StretchHold, TouchIndex, TouchNose, each recorded on both wrists.

<!-- TODO: add the dataset source/citation and a short description of the preprocessing steps (segmentation, QC thresholds, padding). -->
