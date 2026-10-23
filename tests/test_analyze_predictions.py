"""Tests for analyze_predictions.py functions."""

import json
import os
import subprocess
import sys

import pytest

from analyze_predictions import (
    TIER_GOOD,
    TIER_LOW,
    TIER_SUSPICIOUS,
    classify_tier,
    compute_margin,
    evaluate_record,
    find_record,
    load_predictions,
    summarize,
)

THRESHOLDS = dict(low_confidence_threshold=0.30, good_confidence_threshold=0.60, good_margin_threshold=0.35)


def make_record(file_path, predicted_family, top5):
    return {
        "file_path": file_path,
        "predicted_family": predicted_family,
        "confidence": top5[0]["probability"],
        "top5": top5,
    }


# ---------------------------------------------------------------------------
# compute_margin
# ---------------------------------------------------------------------------

class TestComputeMargin:
    def test_normal_top5(self):
        top5 = [{"family": "a", "probability": 0.7}, {"family": "b", "probability": 0.2}]
        assert compute_margin(top5) == pytest.approx(0.5)

    def test_single_entry_margin_equals_confidence(self):
        top5 = [{"family": "a", "probability": 0.9}]
        assert compute_margin(top5) == pytest.approx(0.9)

    def test_empty_top5_returns_zero(self):
        assert compute_margin([]) == 0.0


# ---------------------------------------------------------------------------
# classify_tier
# ---------------------------------------------------------------------------

class TestClassifyTier:
    def test_low_confidence_floor(self):
        # Huge margin but below the low-confidence floor
        assert classify_tier(0.20, 0.95, **THRESHOLDS) == TIER_LOW

    def test_good_match_confidence_and_margin_pass(self):
        assert classify_tier(0.70, 0.40, **THRESHOLDS) == TIER_GOOD

    def test_high_confidence_low_margin_is_suspicious(self):
        # Passes confidence bar but the runner-up is close — combined AND logic
        assert classify_tier(0.65, 0.10, **THRESHOLDS) == TIER_SUSPICIOUS

    def test_moderate_confidence_is_suspicious(self):
        assert classify_tier(0.45, 0.40, **THRESHOLDS) == TIER_SUSPICIOUS

    def test_boundary_values_are_inclusive(self):
        assert classify_tier(0.60, 0.35, **THRESHOLDS) == TIER_GOOD
        assert classify_tier(0.30, 0.0, **THRESHOLDS) != TIER_LOW


# ---------------------------------------------------------------------------
# evaluate_record
# ---------------------------------------------------------------------------

class TestEvaluateRecord:
    def test_adds_derived_fields(self):
        record = make_record("f1", "mirai", [
            {"family": "mirai", "probability": 0.8},
            {"family": "gafgyt", "probability": 0.1},
        ])
        result = evaluate_record(record, **THRESHOLDS)
        assert result["margin"] == pytest.approx(0.7)
        assert result["tier"] == TIER_GOOD
        assert result["second_family"] == "gafgyt"
        assert result["second_probability"] == pytest.approx(0.1)

    def test_missing_confidence_falls_back_to_top5(self):
        record = {"file_path": "f1", "predicted_family": "mirai",
                   "top5": [{"family": "mirai", "probability": 0.5}]}
        result = evaluate_record(record, **THRESHOLDS)
        assert result["confidence"] == pytest.approx(0.5)

    def test_missing_top5_does_not_raise(self):
        record = {"file_path": "f1", "predicted_family": "mirai", "confidence": 0.1}
        result = evaluate_record(record, **THRESHOLDS)
        assert result["tier"] == TIER_LOW
        assert result["second_family"] is None


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------

class TestLoadPredictions:
    def test_loads_valid_file(self, tmp_path):
        path = tmp_path / "preds.json"
        path.write_text(json.dumps([make_record("f1", "mirai", [{"family": "mirai", "probability": 0.9}])]))
        records = load_predictions(str(path))
        assert isinstance(records, list)
        assert len(records) == 1

    def test_non_array_root_exits(self, tmp_path):
        path = tmp_path / "preds.json"
        path.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(SystemExit):
            load_predictions(str(path))

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_predictions(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_exits(self, tmp_path):
        path = tmp_path / "preds.json"
        path.write_text("{not valid json")
        with pytest.raises(SystemExit):
            load_predictions(str(path))


# ---------------------------------------------------------------------------
# find_record
# ---------------------------------------------------------------------------

class TestFindRecord:
    def setup_method(self):
        self.records = [
            make_record("abc123", "mirai", [{"family": "mirai", "probability": 0.9}]),
            make_record("def456", "gafgyt", [{"family": "gafgyt", "probability": 0.8}]),
        ]

    def test_exact_id_match(self):
        result = find_record(self.records, id_="abc123")
        assert result["file_path"] == "abc123"

    def test_case_insensitive_fallback(self):
        result = find_record(self.records, id_="ABC123")
        assert result["file_path"] == "abc123"

    def test_index_lookup(self):
        result = find_record(self.records, index=1)
        assert result["file_path"] == "def456"

    def test_index_out_of_range_exits(self):
        with pytest.raises(SystemExit):
            find_record(self.records, index=99)

    def test_id_not_found_exits(self):
        with pytest.raises(SystemExit):
            find_record(self.records, id_="nonexistent")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def _evaluated(self):
        records = [
            make_record("f1", "mirai", [{"family": "mirai", "probability": 0.9}, {"family": "gafgyt", "probability": 0.05}]),
            make_record("f2", "Mirai", [{"family": "Mirai", "probability": 0.85}, {"family": "gafgyt", "probability": 0.02}]),
            make_record("f3", "gafgyt", [{"family": "gafgyt", "probability": 0.5}, {"family": "mirai", "probability": 0.45}]),
            make_record("f4", "tsunami", [{"family": "tsunami", "probability": 0.1}]),
        ]
        evaluated = []
        for r in records:
            e = evaluate_record(r, **THRESHOLDS)
            e["record"] = r
            evaluated.append(e)
        return evaluated

    def test_tier_counts_sum_to_total(self):
        summary = summarize(self._evaluated())
        assert sum(summary["tier_counts"].values()) == summary["total"] == 4

    def test_case_insensitive_family_grouping(self):
        summary = summarize(self._evaluated())
        # "mirai" and "Mirai" should be grouped together under good_match
        good_counter = summary["family_by_tier"][TIER_GOOD]
        assert good_counter.get("mirai") == 2

    def test_ambiguous_pairs_recorded_for_suspicious(self):
        summary = summarize(self._evaluated())
        assert sum(summary["ambiguous_pairs"].values()) >= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analyze_predictions.py")


class TestAnalyzeCLI:
    def _run(self, args, stdin=None):
        return subprocess.run(
            [sys.executable, SCRIPT] + args,
            input=stdin, capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

    def _write_fixture(self, tmp_path):
        records = [
            make_record("abc123", "mirai", [
                {"family": "mirai", "probability": 0.9},
                {"family": "gafgyt", "probability": 0.05},
            ]),
            make_record("def456", "gafgyt", [
                {"family": "gafgyt", "probability": 0.5},
                {"family": "mirai", "probability": 0.45},
            ]),
            make_record("ghi789", "tsunami", [
                {"family": "tsunami", "probability": 0.1},
            ]),
        ]
        path = tmp_path / "predictions.json"
        path.write_text(json.dumps(records))
        return path

    def test_single_mode_by_id(self, tmp_path):
        path = self._write_fixture(tmp_path)
        result = self._run(["--input", str(path), "--id", "abc123"])
        assert result.returncode == 0, result.stderr
        assert "mirai" in result.stdout
        assert "GOOD MATCH" in result.stdout

    def test_single_mode_by_index(self, tmp_path):
        path = self._write_fixture(tmp_path)
        result = self._run(["--input", str(path), "--index", "0"])
        assert result.returncode == 0, result.stderr
        assert "abc123" in result.stdout

    def test_id_and_index_mutually_exclusive(self, tmp_path):
        path = self._write_fixture(tmp_path)
        result = self._run(["--input", str(path), "--id", "abc123", "--index", "0"])
        assert result.returncode != 0

    def test_bulk_mode_reports_all_tiers(self, tmp_path):
        path = self._write_fixture(tmp_path)
        result = self._run(["--input", str(path)])
        assert result.returncode == 0, result.stderr
        assert "Good match" in result.stdout
        assert "Suspicious" in result.stdout
        assert "Low-confidence" in result.stdout
        assert "Loaded 3 records" in result.stdout

    def test_csv_output(self, tmp_path):
        path = self._write_fixture(tmp_path)
        out_path = tmp_path / "out.csv"
        result = self._run(["--input", str(path), "--csv-output", str(out_path)])
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 4  # header + 3 rows
        assert "tier" in lines[0]

    def test_json_output(self, tmp_path):
        path = self._write_fixture(tmp_path)
        out_path = tmp_path / "out.json"
        result = self._run(["--input", str(path), "--json-output", str(out_path)])
        assert result.returncode == 0, result.stderr
        records = json.loads(out_path.read_text())
        assert len(records) == 3
        for r in records:
            assert "tier" in r
            assert "margin" in r

    def test_custom_thresholds_change_tier(self, tmp_path):
        path = self._write_fixture(tmp_path)
        out_path = tmp_path / "out.json"
        result = self._run([
            "--input", str(path), "--json-output", str(out_path),
            "--good-confidence-threshold", "0.99",
        ])
        assert result.returncode == 0, result.stderr
        records = json.loads(out_path.read_text())
        by_id = {r["file_path"]: r for r in records}
        assert by_id["abc123"]["tier"] == TIER_SUSPICIOUS

    def test_malformed_record_does_not_crash(self, tmp_path):
        records = [{"file_path": "broken", "predicted_family": "mirai", "confidence": 0.5}]
        path = tmp_path / "predictions.json"
        path.write_text(json.dumps(records))
        result = self._run(["--input", str(path)])
        assert result.returncode == 0, result.stderr
        assert "Loaded 1 records" in result.stdout

    def test_missing_input_file_exits_nonzero(self, tmp_path):
        result = self._run(["--input", str(tmp_path / "nonexistent.json")])
        assert result.returncode != 0
