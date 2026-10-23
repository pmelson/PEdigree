"""Tests for pe_features/strings.py."""

from pe_features.strings import (
    count_patterns,
    count_wide_patterns,
    extract_strings,
    extract_wide_strings,
)


# ---------------------------------------------------------------------------
# extract_strings
# ---------------------------------------------------------------------------

def test_extract_strings_basic():
    data = b"Hello, world!"
    results = extract_strings(data, min_len=6)
    assert "Hello, world!" in results


def test_extract_strings_min_len_filter():
    # 5 printable chars should be excluded with default min_len=6
    data = b"short" + b"\x00" + b"long enough string"
    results = extract_strings(data, min_len=6)
    assert "short" not in results
    assert "long enough string" in results


def test_extract_strings_default_min_len_is_six():
    data = b"fiveX" + b"\x00" + b"sixstr"
    results = extract_strings(data)
    assert "fiveX" not in results
    assert "sixstr" in results


def test_extract_strings_empty_input():
    assert extract_strings(b"") == []


def test_extract_strings_non_ascii_excluded():
    data = b"\x80\x90\xffHello World"
    results = extract_strings(data, min_len=6)
    assert all("\x80" not in s for s in results)
    assert "Hello World" in results


# ---------------------------------------------------------------------------
# extract_wide_strings
# ---------------------------------------------------------------------------

def test_extract_wide_strings_basic():
    text = "Hello"
    wide = text.encode("utf-16-le")
    results = extract_wide_strings(wide, min_len=4)
    assert any("Hello" in s for s in results)


def test_extract_wide_strings_empty_input():
    assert extract_wide_strings(b"") == []


def test_extract_wide_strings_round_trip():
    text = "VirtualAlloc"
    wide = text.encode("utf-16-le")
    results = extract_wide_strings(wide, min_len=4)
    assert results and results[0] == text


# ---------------------------------------------------------------------------
# count_patterns
# ---------------------------------------------------------------------------

def test_count_patterns_empty_list():
    result = count_patterns([])
    assert result["num_strings"] == 0
    assert result["num_url_strings"] == 0
    assert result["avg_string_length"] == 0.0


def test_count_patterns_url_detection():
    strings = ["http://evil.com/payload", "https://c2.example.org", "ftp://files.net"]
    result = count_patterns(strings)
    assert result["num_url_strings"] == 3


def test_count_patterns_ip_detection():
    strings = ["Connect to 192.168.1.1 for c2", "invalid.999.0.0", "10.0.0.1"]
    result = count_patterns(strings)
    assert result["num_ip_strings"] == 2


def test_count_patterns_registry_detection():
    strings = [r"HKLM\Software\evil", r"HKCU\Run\malware", "not a registry key"]
    result = count_patterns(strings)
    assert result["num_registry_strings"] == 2


def test_count_patterns_filepath_detection():
    strings = [r"C:\Windows\system32\cmd.exe", "%APPDATA%\\malware.exe"]
    result = count_patterns(strings)
    assert result["num_filepath_strings"] == 2


def test_count_patterns_base64_detection():
    # 40+ chars of valid base64
    b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    result = count_patterns([b64, "short"])
    assert result["num_base64_strings"] == 1


def test_count_patterns_api_like_detection():
    strings = ["VirtualAlloc memory region", "CreateFile handle", "unrelated stuff"]
    result = count_patterns(strings)
    assert result["num_api_like_strings"] == 2


def test_count_patterns_num_strings_and_avg_length():
    strings = ["abcdef", "ghijklmno"]  # lengths 6, 9
    result = count_patterns(strings)
    assert result["num_strings"] == 2
    assert abs(result["avg_string_length"] - 7.5) < 1e-9


# ---------------------------------------------------------------------------
# count_wide_patterns
# ---------------------------------------------------------------------------

def test_count_wide_patterns_uses_wide_prefix():
    strings = ["http://evil.com/payload"]
    result = count_wide_patterns(strings)
    assert "num_wide_strings" in result
    assert "num_wide_url_strings" in result
    assert "avg_wide_string_length" in result
    assert result["num_wide_url_strings"] == 1


def test_count_wide_patterns_empty_list():
    result = count_wide_patterns([])
    assert result["num_wide_strings"] == 0
    assert result["avg_wide_string_length"] == 0.0
