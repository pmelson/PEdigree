"""Tests for extract_features.py functions and CLI."""

import csv
import json
import os
import subprocess
import sys

import pytest

# Import functions directly
from extract_features import _collect_files, _extract_one, _write_csv, _write_jsonl
from pathlib import Path


# ---------------------------------------------------------------------------
# _collect_files
# ---------------------------------------------------------------------------

class TestCollectFiles:
    def test_single_file(self, pe32_file):
        result = _collect_files(pe32_file, recursive=False)
        assert result == [pe32_file]

    def test_flat_dir(self, tmp_path, pe32_file):
        result = _collect_files(tmp_path, recursive=False)
        assert pe32_file in result

    def test_recursive_dir(self, tmp_path, minimal_pe32_bytes):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        child = subdir / "nested.exe"
        child.write_bytes(minimal_pe32_bytes)
        top = tmp_path / "top.exe"
        top.write_bytes(minimal_pe32_bytes)
        non_recursive = _collect_files(tmp_path, recursive=False)
        recursive = _collect_files(tmp_path, recursive=True)
        assert len(recursive) > len(non_recursive)
        assert child in recursive

    def test_nonexistent_path_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _collect_files(tmp_path / "no_such_dir", recursive=False)


# ---------------------------------------------------------------------------
# _extract_one
# ---------------------------------------------------------------------------

class TestExtractOne:
    def test_returns_dict(self, pe32_file):
        result = _extract_one((str(pe32_file), None))
        assert isinstance(result, dict)

    def test_label_attached_when_given(self, pe32_file):
        result = _extract_one((str(pe32_file), "mirai"))
        assert result["label"] == "mirai"

    def test_no_label_key_when_none(self, pe32_file):
        result = _extract_one((str(pe32_file), None))
        assert "label" not in result


# ---------------------------------------------------------------------------
# _write_jsonl / _write_csv
# ---------------------------------------------------------------------------

class TestWriteFormats:
    def test_write_jsonl_creates_file(self, tmp_path):
        out = tmp_path / "out.jsonl"
        features = [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}]
        count = _write_jsonl(iter(features), out)
        assert out.exists()
        assert count == 2

    def test_write_jsonl_valid_json_lines(self, tmp_path):
        out = tmp_path / "out.jsonl"
        features = [{"x": 1}, {"x": 2}]
        _write_jsonl(iter(features), out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "x" in obj

    def test_write_csv_creates_file(self, tmp_path):
        out = tmp_path / "out.csv"
        features = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        count = _write_csv(iter(features), out)
        assert out.exists()
        assert count == 2

    def test_write_csv_has_header(self, tmp_path):
        out = tmp_path / "out.csv"
        features = [{"col1": 10, "col2": 20}]
        _write_csv(iter(features), out)
        with out.open() as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "col1" in header
        assert "col2" in header


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "extract_features.py")


class TestCLI:
    def test_single_file_jsonl_output(self, tmp_path, pe32_file):
        out = tmp_path / "out.jsonl"
        result = subprocess.run(
            [sys.executable, SCRIPT, str(pe32_file), "-o", str(out), "--format", "jsonl"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert out.exists()
        obj = json.loads(out.read_text().strip().splitlines()[0])
        assert "file_path" in obj

    def test_single_file_csv_output(self, tmp_path, pe32_file):
        out = tmp_path / "out.csv"
        result = subprocess.run(
            [sys.executable, SCRIPT, str(pe32_file), "-o", str(out), "--format", "csv"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert out.exists()
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_label_attached_in_output(self, tmp_path, pe32_file):
        out = tmp_path / "out.jsonl"
        subprocess.run(
            [sys.executable, SCRIPT, str(pe32_file), "-o", str(out), "--label", "testfam"],
            capture_output=True, text=True, check=True,
        )
        obj = json.loads(out.read_text().strip())
        assert obj["label"] == "testfam"

    def test_nonexistent_input_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, SCRIPT, str(tmp_path / "ghost"), "-o", str(tmp_path / "x.jsonl")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
