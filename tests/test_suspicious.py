"""Tests for pe_features/suspicious.py."""

from pe_features.suspicious import HIGH_RISK_APIS, SUSPICIOUS_DLLS


def test_suspicious_dlls_is_frozenset():
    assert isinstance(SUSPICIOUS_DLLS, frozenset)


def test_suspicious_dlls_all_lowercase():
    for dll in SUSPICIOUS_DLLS:
        assert dll == dll.lower(), f"Not lowercase: {dll}"


def test_ws2_32_in_suspicious_dlls():
    assert "ws2_32.dll" in SUSPICIOUS_DLLS


def test_high_risk_apis_is_dict():
    assert isinstance(HIGH_RISK_APIS, dict)


def test_high_risk_apis_all_lowercase_keys():
    for key in HIGH_RISK_APIS:
        assert key == key.lower(), f"Not lowercase key: {key}"


def test_high_risk_apis_all_lowercase_values():
    for val in HIGH_RISK_APIS.values():
        assert val == val.lower(), f"Not lowercase value: {val}"


def test_key_apis_present():
    for api in ("virtualalloc", "createremotethread", "loadlibrary", "sleep"):
        assert api in HIGH_RISK_APIS, f"Missing API: {api}"
