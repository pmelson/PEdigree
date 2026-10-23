"""
PE Malware Family Classifier — Hyperparameter tuning with Optuna.

Requires that train.py has already been run so the preprocessed split data
can be re-created from raw files.  The best model is saved alongside the
baseline model.

Usage:
    python tune.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
                   [--n-trials N] [--timeout SECONDS]
"""

import argparse
import os
import warnings

import joblib
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Reuse helpers from train.py
from train import (
    DATA_DIR,
    OUTPUT_DIR,
    load_data,
    make_splits,
    preprocess,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    return objective


def main():
    parser = argparse.ArgumentParser(description="Tune PE malware classifier with Optuna")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials (default 50)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("PE Malware Classifier — Hyperparameter Tuning")
    print("=" * 60)

    data = load_data(args.data_dir)
    X, y_raw, feature_cols, cat_encoders = preprocess(data)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw)

    X_train, X_val, X_test, y_train, y_val, y_test_enc, y_test_raw = make_splits(
        X, y_raw, label_encoder
    )

    print(f"\nRunning {args.n_trials} Optuna trials ...")
    study = optuna.create_study(direction="maximize", study_name="lgbm_malware")
    study.optimize(
        build_objective(X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    print(f"\nBest val macro-F1: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain on train+val with best params
    print("\nRetraining best model on train+val ...")
    import numpy as np

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    best_params = {**study.best_params, "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1}
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_trainval, y_trainval)

    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test_enc, y_pred, average="macro")
    print(f"Test macro-F1 (tuned model): {test_f1:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    artifact = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "cat_encoders": cat_encoders,
        "best_params": study.best_params,
        "val_macro_f1": study.best_value,
        "test_macro_f1": test_f1,
    }
    tuned_path = os.path.join(args.output_dir, "malware_classifier_tuned.pkl")
    joblib.dump(artifact, tuned_path)
    print(f"Saved tuned model: {tuned_path}")

    # Save study results as CSV
    trials_path = os.path.join(args.output_dir, "optuna_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"Saved trials: {trials_path}")


if __name__ == "__main__":
    main()
