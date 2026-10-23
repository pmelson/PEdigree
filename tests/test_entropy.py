"""Tests for pe_features/entropy.py."""

import math
from unittest.mock import MagicMock

import pytest

from pe_features.entropy import section_entropy, shannon_entropy


def test_empty_bytes_returns_zero():
    assert shannon_entropy(b"") == 0.0


def test_single_repeated_byte_zero_entropy():
    assert shannon_entropy(b"\x00" * 1000) == 0.0


def test_two_values_equal_prob_entropy_one():
    data = b"\x00\xff" * 500
    result = shannon_entropy(data)
    assert abs(result - 1.0) < 1e-9


def test_uniform_all_256_max_entropy():
    data = bytes(range(256))
    result = shannon_entropy(data)
    assert abs(result - 8.0) < 1e-9


def test_returns_float():
    assert isinstance(shannon_entropy(b"\xaa\xbb"), float)


def test_bounds_random_data():
    import os
    data = os.urandom(512)
    result = shannon_entropy(data)
    assert 0.0 <= result <= 8.0


def test_section_entropy_delegates():
    mock_section = MagicMock()
    mock_section.get_data.return_value = b"\x00\xff" * 500
    mock_pe = MagicMock()
    result = section_entropy(mock_section, mock_pe)
    assert abs(result - 1.0) < 1e-9


def test_section_entropy_on_error_returns_zero():
    mock_section = MagicMock()
    mock_section.get_data.side_effect = RuntimeError("no data")
    mock_pe = MagicMock()
    assert section_entropy(mock_section, mock_pe) == 0.0
