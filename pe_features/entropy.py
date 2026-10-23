"""Shannon entropy utilities for PE feature extraction."""

import math


def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy of a byte sequence.

    Returns a value in [0.0, 8.0]. Returns 0.0 for empty input.
    """
    if not data:
        return 0.0

    counts = [0] * 256
    for byte in data:
        counts[byte] += 1

    length = len(data)
    entropy = 0.0
    for count in counts:
        if count:
            p = count / length
            entropy -= p * math.log2(p)

    return entropy


def section_entropy(section, pe) -> float:
    """Compute entropy of a PE section's raw data.

    Args:
        section: A pefile SectionStructure instance.
        pe: The parent pefile.PE instance (used to retrieve raw data).

    Returns:
        Shannon entropy of the section data, or 0.0 if data is unavailable.
    """
    try:
        data = section.get_data()
        return shannon_entropy(data) if data else 0.0
    except Exception:
        return 0.0
