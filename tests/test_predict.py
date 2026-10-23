"""Tests for predict.py functions."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from predict import _cell_to_str, load_model, predict_sample, preprocess_sample


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_loads_existing_pkl(self, trained_artifact):
        artifact, pkl_path = trained_artifact
        loaded = load_model(pkl_path)
        assert "model" in loaded
        assert "label_encoder" in loaded
        assert "feature_cols" in loaded

    def test_model_path_stored_in_artifact(self, trained_artifact):
        _, pkl_path = trained_artifact
        loaded = load_model(pkl_path)
        assert loaded["_model_path"] == pkl_path

    def test_exits_when_model_not_found(self, tmp_path):
        missing = str(tmp_path / "nonexistent.pkl")
        with pytest.raises(SystemExit):
            load_model(missing)

    def test_fallback_to_standard_pkl(self, trained_artifact, tmp_path, monkeypatch):
        """When requested path is DEFAULT_MODEL and it doesn't exist, fall back."""
        import predict as predict_module
        artifact, std_path = trained_artifact
        monkeypatch.setattr(predict_module, "DEFAULT_MODEL", str(tmp_path / "missing_tuned.pkl"))
        monkeypatch.setattr(predict_module, "FALLBACK_MODEL", std_path)
        loaded = load_model(str(tmp_path / "missing_tuned.pkl"))
        assert "model" in loaded


# ---------------------------------------------------------------------------
# _cell_to_str
# ---------------------------------------------------------------------------

class TestCellToStr:
    def test_scalar_string(self):
        assert _cell_to_str("hello") == "hello"

    def test_nan_returns_unknown(self):
        import math
        assert _cell_to_str(float("nan")) == "unknown"

    def test_none_returns_unknown(self):
        assert _cell_to_str(None) == "unknown"

    def test_list_returns_string(self):
        result = _cell_to_str([1, 2, 3])
        assert isinstance(result, str)

    def test_numpy_array_returns_string(self):
        result = _cell_to_str(np.array([1, 2]))
        assert isinstance(result, str)

    def test_integer_returns_string(self):
        assert _cell_to_str(42) == "42"


# ---------------------------------------------------------------------------
# preprocess_sample
# ---------------------------------------------------------------------------

class TestPreprocessSample:
    def test_output_shape(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        X = preprocess_sample(sample_feature_row, artifact)
        assert X.shape == (1, len(artifact["feature_cols"]))

    def test_missing_features_filled_minus_one(self, trained_artifact):
        artifact, _ = trained_artifact
        # Provide an empty sample — all features missing
        X = preprocess_sample({}, artifact)
        assert X.shape[1] == len(artifact["feature_cols"])
        # All values should be -1 (NaN → -1)
        assert (X == -1).all()

    def test_unseen_categorical_returns_minus_one(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        row = sample_feature_row.copy()
        if "machine_type" in artifact["feature_cols"]:
            row["machine_type"] = "UNSEEN_VALUE_XYZ"
        X = preprocess_sample(row, artifact)
        if "machine_type" in artifact["feature_cols"]:
            idx = artifact["feature_cols"].index("machine_type")
            assert X[0, idx] == -1


# ---------------------------------------------------------------------------
# predict_sample
# ---------------------------------------------------------------------------

class TestPredictSample:
    def test_returns_expected_keys(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        result = predict_sample(sample_feature_row, artifact)
        assert "predicted_family" in result
        assert "confidence" in result
        assert "top5" in result

    def test_confidence_in_valid_range(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        result = predict_sample(sample_feature_row, artifact)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top5_max_five_entries(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        result = predict_sample(sample_feature_row, artifact)
        assert len(result["top5"]) <= 5

    def test_top5_each_has_family_and_probability(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        result = predict_sample(sample_feature_row, artifact)
        for entry in result["top5"]:
            assert "family" in entry
            assert "probability" in entry
            assert 0.0 <= entry["probability"] <= 1.0

    def test_predicted_family_is_known_label(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        result = predict_sample(sample_feature_row, artifact)
        known = set(artifact["label_encoder"].classes_)
        assert result["predicted_family"] in known

    def test_top5_probabilities_sum_to_one_or_less(self, trained_artifact, sample_feature_row):
        artifact, _ = trained_artifact
        result = predict_sample(sample_feature_row, artifact)
        total = sum(e["probability"] for e in result["top5"])
        assert total <= 1.001  # floating point tolerance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "predict.py")


class TestPredictCLI:
    def _run(self, args, stdin=None, cwd=None):
        return subprocess.run(
            [sys.executable, SCRIPT] + args,
            input=stdin, capture_output=True, text=True,
            cwd=cwd or os.path.dirname(os.path.dirname(__file__)),
        )

    def test_file_input(self, trained_artifact, tmp_path, sample_feature_row):
        artifact, pkl_path = trained_artifact
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample_feature_row))
        result = self._run(["--input", str(sample_path), "--model", pkl_path])
        assert result.returncode == 0, result.stderr

    def test_stdin_input(self, trained_artifact, tmp_path, sample_feature_row):
        artifact, pkl_path = trained_artifact
        result = self._run(
            ["--model", pkl_path],
            stdin=json.dumps(sample_feature_row),
        )
        assert result.returncode == 0, result.stderr

    def test_json_output_flag(self, trained_artifact, tmp_path, sample_feature_row):
        artifact, pkl_path = trained_artifact
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample_feature_row))
        out_path = tmp_path / "results.json"
        result = self._run([
            "--input", str(sample_path),
            "--model", pkl_path,
            "--json-output", str(out_path),
        ])
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        records = json.loads(out_path.read_text())
        assert isinstance(records, list)
        assert len(records) == 1
        assert "predicted_family" in records[0]
        assert "confidence" in records[0]
