# tune.py

Optimize LightGBM hyperparameters using Optuna and save a tuned model artifact.

## Overview

`tune.py` loads the same feature data as `train.py`, reconstructs the identical train/validation/test split, and runs an Optuna study to maximize macro-F1 on the validation set. After the search, the best hyperparameters are used to retrain a final model on the combined train+validation data, which is then evaluated on the held-out test set and saved.

`predict.py` automatically prefers the tuned artifact (`malware_classifier_tuned.pkl`) over the baseline artifact from `train.py`.

## Prerequisites

- The same `.json` NDJSON feature files used by `train.py` — `tune.py` reads and preprocesses the data itself.
- `train.py` does **not** need to have been run first.
- Optuna must be installed (`pip install optuna`).

## Usage

```bash
# Run 50 trials (default) with the same data directory used for training
python tune.py --data-dir ./data/ --output-dir ./models/run1/

# Limit by number of trials
python tune.py --data-dir ./data/ --output-dir ./output/ --n-trials 100

# Limit by wall-clock time instead (seconds)
python tune.py --data-dir ./data/ --output-dir ./output/ --timeout 3600

# Both limits — whichever is reached first
python tune.py --data-dir ./data/ --output-dir ./output/ --n-trials 200 --timeout 7200
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | Script directory | Directory containing `.json` NDJSON family files |
| `--output-dir` | `./output/` | Directory where output artifacts are written |
| `--n-trials` | `50` | Maximum number of Optuna trials |
| `--timeout` | *(none)* | Stop after this many seconds regardless of trial count |

## Hyperparameter Search Space

| Parameter | Type | Range |
|---|---|---|
| `n_estimators` | int | 200 – 1200 |
| `learning_rate` | float (log scale) | 0.01 – 0.3 |
| `num_leaves` | int | 31 – 255 |
| `min_child_samples` | int | 5 – 60 |
| `subsample` | float | 0.5 – 1.0 |
| `colsample_bytree` | float | 0.5 – 1.0 |
| `reg_alpha` | float (log scale) | 1e-8 – 10.0 |
| `reg_lambda` | float (log scale) | 1e-8 – 10.0 |

Each trial also uses `class_weight="balanced"` and `random_state=42` (fixed, not tuned).

## Outputs

All files are written to `--output-dir`:

| File | Description |
|---|---|
| `malware_classifier_tuned.pkl` | Tuned model artifact (see below) |
| `optuna_trials.csv` | Full trial history: parameters and validation F1 for every trial |

### Tuned model artifact (`malware_classifier_tuned.pkl`)

Contains all keys from the baseline artifact plus tuning-specific fields:

| Key | Type | Description |
|---|---|---|
| `model` | `LGBMClassifier` | Final model trained on train+val with best params |
| `label_encoder` | `LabelEncoder` | Maps family name strings ↔ integer class indices |
| `feature_cols` | `list[str]` | Ordered feature column names |
| `cat_encoders` | `dict` | Per-column categorical encoders |
| `best_params` | `dict` | Winning hyperparameter values from the Optuna study |
| `val_macro_f1` | `float` | Best validation macro-F1 achieved during the search |
| `test_macro_f1` | `float` | Macro-F1 of the final model on the held-out test set |

## Notes

- Runtime scales roughly linearly with `--n-trials`. Each trial trains a full LightGBM model with early stopping (30 rounds); 50 trials typically completes in a few minutes on a modern machine.
- `optuna_trials.csv` can be used to visualize the search landscape or inspect which parameter regions performed best.
- The final model is trained on **train + validation** combined, giving it more data than the per-trial models. The `test_macro_f1` in the artifact reflects this final, larger model.
- Because `tune.py` uses the same random seed (42) for data splitting as `train.py`, the held-out test set is identical between the two scripts — their test metrics are directly comparable.
- Optuna logging is suppressed to WARNING level by default. Add `optuna.logging.set_verbosity(optuna.logging.INFO)` to see per-trial progress beyond the built-in progress bar.
