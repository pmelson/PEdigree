"""Printable string extraction and pattern matching for PE feature extraction."""

import re

# Pattern constants for classification
_URL_RE = re.compile(r"https?://|ftp://", re.IGNORECASE)
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_REGISTRY_RE = re.compile(r"HKEY_|HKLM\\|HKCU\\|HKCR\\|HKU\\", re.IGNORECASE)
_FILEPATH_RE = re.compile(
    r"[A-Za-z]:\\|%(?:APPDATA|TEMP|WINDIR|SYSTEMROOT|PROGRAMFILES|USERPROFILE)%",
    re.IGNORECASE,
)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

_API_FRAGMENTS: frozenset[str] = frozenset({
    "alloc", "process", "thread", "inject", "hook", "registry",
    "connect", "socket", "recv", "send", "encrypt", "decrypt",
    "createfile", "writefile", "readfile", "loadlibrary",
    "getprocaddress", "virtualalloc", "writeprocessmemory",
})


def extract_strings(data: bytes, min_len: int = 6) -> list[str]:
    """Extract printable ASCII runs from raw bytes."""
    pattern = re.compile(rb"[ -~]{%d,}" % min_len)
    return [m.group().decode("ascii", errors="replace") for m in pattern.finditer(data)]


def extract_wide_strings(data: bytes, min_len: int = 4) -> list[str]:
    """Extract UTF-16LE strings from raw bytes.

    Matches runs of (printable ASCII byte, 0x00) pairs, which covers the
    vast majority of wide strings in .NET assemblies and native Windows
    binaries. Each match is decoded as UTF-16LE.

    Args:
        data: Raw byte content to scan.
        min_len: Minimum number of wide characters to include (default 4).

    Returns:
        List of decoded UTF-16LE strings.
    """
    pattern = re.compile(rb"(?:[\x20-\x7e]\x00){%d,}" % min_len)
    results = []
    for m in pattern.finditer(data):
        try:
            decoded = m.group().decode("utf-16-le", errors="replace")
            results.append(decoded)
        except Exception:
            pass
    return results


def count_patterns(strings: list[str]) -> dict[str, int | float]:
    """Count occurrences of suspicious string patterns.

    Args:
        strings: List of printable strings extracted from a PE file.

    Returns:
        Dict with counts for each pattern category and aggregate stats.
        Keys are prefixed with ``num_`` (and ``avg_`` for the length mean).
        Pass wide=True strings here via the separate count_wide_patterns()
        call to get ``num_wide_*`` keys instead.
    """
    return _count(strings, prefix="")


def count_wide_patterns(strings: list[str]) -> dict[str, int | float]:
    """Count suspicious patterns in UTF-16LE (wide) strings.

    Produces the same categories as count_patterns() but under
    ``num_wide_*`` / ``avg_wide_string_length`` keys.

    Args:
        strings: List of decoded wide strings from extract_wide_strings().

    Returns:
        Dict with ``num_wide_*`` and ``avg_wide_string_length`` keys.
    """
    return _count(strings, prefix="wide_")


def _count(strings: list[str], prefix: str) -> dict[str, int | float]:
    empty: dict[str, int | float] = {
        f"num_{prefix}strings": 0,
        f"avg_{prefix}string_length": 0.0,
        f"num_{prefix}url_strings": 0,
        f"num_{prefix}ip_strings": 0,
        f"num_{prefix}registry_strings": 0,
        f"num_{prefix}filepath_strings": 0,
        f"num_{prefix}base64_strings": 0,
        f"num_{prefix}api_like_strings": 0,
    }
    if not strings:
        return empty

    lengths = [len(s) for s in strings]
    counts: dict[str, int | float] = {
        f"num_{prefix}strings": len(strings),
        f"avg_{prefix}string_length": sum(lengths) / len(lengths),
        f"num_{prefix}url_strings": 0,
        f"num_{prefix}ip_strings": 0,
        f"num_{prefix}registry_strings": 0,
        f"num_{prefix}filepath_strings": 0,
        f"num_{prefix}base64_strings": 0,
        f"num_{prefix}api_like_strings": 0,
    }

    lower_fragments = {f.lower() for f in _API_FRAGMENTS}

    for s in strings:
        if _URL_RE.search(s):
            counts[f"num_{prefix}url_strings"] += 1  # type: ignore[operator]
        if _IP_RE.search(s):
            counts[f"num_{prefix}ip_strings"] += 1  # type: ignore[operator]
        if _REGISTRY_RE.search(s):
            counts[f"num_{prefix}registry_strings"] += 1  # type: ignore[operator]
        if _FILEPATH_RE.search(s):
            counts[f"num_{prefix}filepath_strings"] += 1  # type: ignore[operator]
        if _BASE64_RE.search(s):
            counts[f"num_{prefix}base64_strings"] += 1  # type: ignore[operator]
        s_lower = s.lower()
        if any(frag in s_lower for frag in lower_fragments):
            counts[f"num_{prefix}api_like_strings"] += 1  # type: ignore[operator]

    return counts
