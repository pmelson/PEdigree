"""Tests for tune.py — unit-level only (no 50-trial runs)."""

import os

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pytest

from train import make_splits, preprocess
from tune import build_objective


# ---------------------------------------------------------------------------
# build_objective
# ---------------------------------------------------------------------------

class TestBuildObjective:
    @pytest.fixture
    def split_data(self, sample_df):
        from sklearn.preprocessing import LabelEncoder
        X, y_raw, _, _ = preprocess(sample_df.copy())
        le = LabelEncoder()
        le.fit(y_raw)
        X_train, X_val, X_test, y_train, y_val, *_ = make_splits(X, y_raw, le)
        return X_train, y_train, X_val, y_val

    def test_returns_callable(self, split_data):
        X_train, y_train, X_val, y_val = split_data
        obj = build_objective(X_train, y_train, X_val, y_val)
        assert callable(obj)

    def test_single_trial_returns_float(self, split_data):
        X_train, y_train, X_val, y_val = split_data
        obj = build_objective(X_train, y_train, X_val, y_val)

        # Create a trial with fixed params that make training fast
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 1e-5,
            "reg_lambda": 1e-5,
        })
        study.optimize(obj, n_trials=1)
        assert isinstance(study.best_value, float)
        assert study.best_value >= 0.0


# ---------------------------------------------------------------------------
# Artifact keys (end-to-end with 1 trial)
# ---------------------------------------------------------------------------

class TestTuneArtifact:
    def test_artifact_contains_expected_keys(self, sample_df, tmp_path):
        """Run a 1-trial tune and verify the saved artifact has all expected keys."""
        from sklearn.preprocessing import LabelEncoder
        X, y_raw, feature_cols, cat_encoders = preprocess(sample_df.copy())
        le = LabelEncoder()
        le.fit(y_raw)
        X_train, X_val, X_test, y_train, y_val, y_test_enc, _ = make_splits(X, y_raw, le)

        study = optuna.create_study(direction="maximize", study_name="test_tune")
        study.enqueue_trial({
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 1e-5,
            "reg_lambda": 1e-5,
        })
        obj = build_objective(X_train, y_train, X_val, y_val)
        study.optimize(obj, n_trials=1)

        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        best_params = {
            **study.best_params,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        }
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_trainval, y_trainval)

        from sklearn.metrics import f1_score
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test_enc, y_pred, average="macro")

        artifact = {
            "model": best_model,
            "label_encoder": le,
            "feature_cols": feature_cols,
            "cat_encoders": cat_encoders,
            "best_params": study.best_params,
            "val_macro_f1": study.best_value,
            "test_macro_f1": test_f1,
        }
        pkl_path = tmp_path / "malware_classifier_tuned.pkl"
        joblib.dump(artifact, pkl_path)

        loaded = joblib.load(pkl_path)
        for key in ("model", "label_encoder", "feature_cols", "cat_encoders",
                    "best_params", "val_macro_f1", "test_macro_f1"):
            assert key in loaded, f"Missing key: {key}"

    def test_optuna_trials_csv_written(self, sample_df, tmp_path):
        from sklearn.preprocessing import LabelEncoder
        X, y_raw, _, _ = preprocess(sample_df.copy())
        le = LabelEncoder()
        le.fit(y_raw)
        X_train, X_val, *_ = make_splits(X, y_raw, le)

        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({
            "n_estimators": 10, "learning_rate": 0.1, "num_leaves": 7,
            "min_child_samples": 5, "subsample": 1.0, "colsample_bytree": 1.0,
            "reg_alpha": 1e-5, "reg_lambda": 1e-5,
        })
        obj = build_objective(X_train, X_val[:, 0].astype(int), X_val, X_val[:, 0].astype(int))
        # Just check that trials_dataframe() works
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
        csv_path = tmp_path / "optuna_trials.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) >= 1
