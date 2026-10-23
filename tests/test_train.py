"""Tests for train.py functions."""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from train import (
    CAT_COLS,
    DROP_COLS,
    MIN_SAMPLES_FOR_SPLIT,
    evaluate,
    load_data,
    make_splits,
    plot_feature_importance,
    preprocess,
    train_model,
)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_loads_ndjson_from_dir(self, ndjson_data_dir):
        df = load_data(str(ndjson_data_dir))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_label_column_set_from_filename_when_absent(self, tmp_path):
        row = {"file_size": 100, "file_entropy": 5.0}
        (tmp_path / "myfamily.json").write_text(json.dumps(row) + "\n")
        df = load_data(str(tmp_path))
        assert (df["label"] == "myfamily").all()

    def test_label_column_used_when_present(self, tmp_path):
        row = {"file_size": 100, "label": "customlabel"}
        (tmp_path / "whatever.json").write_text(json.dumps(row) + "\n")
        df = load_data(str(tmp_path))
        assert "customlabel" in df["label"].values

    def test_multiple_files_concatenated(self, ndjson_data_dir):
        df = load_data(str(ndjson_data_dir))
        # 3 families × 30 rows each = 90 total
        assert len(df) == 90

    def test_exits_on_empty_dir(self, tmp_path):
        with pytest.raises(SystemExit):
            load_data(str(tmp_path))


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_returns_four_values(self, sample_df):
        result = preprocess(sample_df.copy())
        assert len(result) == 4

    def test_drop_cols_not_in_X(self, sample_df):
        X, y_raw, feature_cols, _ = preprocess(sample_df.copy())
        for col in DROP_COLS:
            assert col not in feature_cols

    def test_cat_cols_encoded_to_numeric(self, sample_df):
        X, y_raw, feature_cols, _ = preprocess(sample_df.copy())
        # All encoded values should be numeric integers
        for col in CAT_COLS:
            if col in feature_cols:
                idx = feature_cols.index(col)
                vals = X[:, idx]
                assert np.issubdtype(vals.dtype, np.number)

    def test_nan_filled_with_minus_one(self, sample_df):
        df = sample_df.copy()
        # Introduce NaN in a numeric column
        df.loc[0, "file_size"] = np.nan
        X, _, _, _ = preprocess(df)
        feature_cols_check = [
            c for c in df.columns
            if c not in DROP_COLS and c not in CAT_COLS and c != "label"
        ]
        if "file_size" in feature_cols_check:
            idx = feature_cols_check.index("file_size")
            assert X[0, idx] == -1.0

    def test_y_raw_has_family_strings(self, sample_df):
        _, y_raw, _, _ = preprocess(sample_df.copy())
        assert set(y_raw) == {"familyA", "familyB", "familyC"}

    def test_X_is_numpy_array(self, sample_df):
        X, _, _, _ = preprocess(sample_df.copy())
        assert isinstance(X, np.ndarray)

    def test_label_encoders_returned(self, sample_df):
        _, _, _, label_encoders = preprocess(sample_df.copy())
        assert isinstance(label_encoders, dict)


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------

class TestMakeSplits:
    @pytest.fixture
    def splits(self, sample_df):
        from sklearn.preprocessing import LabelEncoder
        X, y_raw, _, _ = preprocess(sample_df.copy())
        le = LabelEncoder()
        le.fit(y_raw)
        return make_splits(X, y_raw, le), X, y_raw

    def test_returns_seven_values(self, splits):
        result, X, y_raw = splits
        assert len(result) == 7

    def test_total_sizes_add_up(self, splits):
        result, X, y_raw = splits
        X_train, X_val, X_test = result[0], result[1], result[2]
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == len(X)

    def test_test_size_roughly_15_percent(self, splits):
        result, X, _ = splits
        X_test = result[2]
        ratio = X_test.shape[0] / len(X)
        assert 0.10 <= ratio <= 0.20

    def test_rare_class_only_in_train(self, sample_df):
        """A class with < MIN_SAMPLES_FOR_SPLIT samples must be train-only."""
        from sklearn.preprocessing import LabelEncoder
        # Add a tiny rare class
        rare_rows = sample_df.iloc[:2].copy()
        rare_rows["label"] = "rare_family"
        df = pd.concat([sample_df, rare_rows], ignore_index=True)
        X, y_raw, _, _ = preprocess(df.copy())
        le = LabelEncoder()
        le.fit(y_raw)
        X_train, X_val, X_test, y_train, y_val, y_test_enc, y_test_raw = make_splits(
            X, y_raw, le
        )
        rare_idx = list(le.classes_).index("rare_family")
        assert rare_idx not in y_val
        assert rare_idx not in y_test_enc

    def test_all_non_rare_classes_in_test(self, splits):
        from sklearn.preprocessing import LabelEncoder
        result, X, y_raw = splits
        _, _, _, _, _, y_test_enc, _ = result
        # All three families should appear in the test set
        le = LabelEncoder()
        le.fit(y_raw)
        assert len(set(y_test_enc)) == 3


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    @pytest.fixture
    def training_data(self, sample_df):
        from sklearn.preprocessing import LabelEncoder
        X, y_raw, _, _ = preprocess(sample_df.copy())
        le = LabelEncoder()
        le.fit(y_raw)
        splits = make_splits(X, y_raw, le)
        return splits[0], splits[1], splits[3], splits[4]  # X_train, X_val, y_train, y_val

    def test_returns_lgbm_classifier(self, training_data):
        X_train, X_val, y_train, y_val = training_data
        model = train_model(X_train, y_train, X_val, y_val)
        assert isinstance(model, lgb.LGBMClassifier)

    def test_predict_proba_shape(self, training_data):
        X_train, X_val, y_train, y_val = training_data
        model = train_model(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_val)
        assert proba.shape[0] == X_val.shape[0]
        assert proba.shape[1] == 3  # 3 families


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    @pytest.fixture
    def eval_setup(self, trained_artifact, tmp_path):
        artifact, _ = trained_artifact
        model = artifact["model"]
        le = artifact["label_encoder"]
        X, y_raw, _, _ = preprocess(
            pd.DataFrame([{c: 0.0 for c in artifact["feature_cols"]}] * 10)
            .assign(label="familyA")
        )
        # Use training data for a quick eval smoke test
        from sklearn.preprocessing import LabelEncoder
        full_le = le
        y_enc = full_le.transform(["familyA"] * 10)
        y_raw_arr = np.array(["familyA"] * 10)
        output_dir = str(tmp_path / "output")
        return model, X, y_enc, y_raw_arr, full_le, output_dir

    def test_creates_classification_report(self, trained_artifact, tmp_path):
        artifact, _ = trained_artifact
        model = artifact["model"]
        le = artifact["label_encoder"]
        output_dir = str(tmp_path / "eval_out")
        X_test = model.booster_.predict(np.zeros((5, len(artifact["feature_cols"]))), raw_score=False)
        # Use the model's training data proxy
        n = 5
        X_dummy = np.zeros((n, len(artifact["feature_cols"])))
        y_enc = np.array([0, 1, 2, 0, 1])
        y_raw = le.inverse_transform(y_enc)
        macro_f1 = evaluate(model, X_dummy, y_enc, y_raw, le, output_dir)
        assert os.path.exists(os.path.join(output_dir, "classification_report.txt"))
        assert isinstance(macro_f1, float)
        assert 0.0 <= macro_f1 <= 1.0

    def test_creates_confusion_matrix_png(self, trained_artifact, tmp_path):
        artifact, _ = trained_artifact
        model = artifact["model"]
        le = artifact["label_encoder"]
        output_dir = str(tmp_path / "eval_out2")
        n = 5
        X_dummy = np.zeros((n, len(artifact["feature_cols"])))
        y_enc = np.array([0, 1, 2, 0, 1])
        y_raw = le.inverse_transform(y_enc)
        evaluate(model, X_dummy, y_enc, y_raw, le, output_dir)
        assert os.path.exists(os.path.join(output_dir, "confusion_matrix.png"))


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:
    def test_creates_png(self, trained_artifact, tmp_path):
        artifact, _ = trained_artifact
        model = artifact["model"]
        output_dir = str(tmp_path / "fi_out")
        os.makedirs(output_dir, exist_ok=True)
        plot_feature_importance(model, artifact["feature_cols"], output_dir)
        assert os.path.exists(os.path.join(output_dir, "feature_importance.png"))
