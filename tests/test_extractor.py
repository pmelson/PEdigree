"""Tests for pe_features/extractor.py (PEFeatureExtractor)."""

import os

import pytest

from pe_features import PEFeatureExtractor
from pe_features.suspicious import HIGH_RISK_APIS

# All numeric/int keys that must appear in every result
_REQUIRED_NUMERIC_KEYS = {
    "file_size", "file_entropy", "has_valid_dos_header",
    "machine_type", "num_sections", "timestamp",
    "timestamp_is_zero", "timestamp_is_future",
    "file_characteristics", "is_dll", "is_exe", "is_64bit",
    "subsystem", "dll_characteristics",
    "has_aslr", "has_dep", "has_no_seh", "has_cfg",
    "size_of_code", "size_of_initialized_data", "size_of_uninitialized_data",
    "size_of_image", "size_of_headers", "address_of_entry_point",
    "image_base", "linker_major_version", "linker_minor_version",
    "os_major_version", "checksum", "checksum_is_valid", "num_data_directories",
    "ratio_code_to_image",
    "num_executable_sections", "num_writable_sections", "num_wx_sections",
    "section_entropy_mean", "section_entropy_max", "section_entropy_min",
    "section_entropy_std", "num_high_entropy_sections",
    "entry_section_entropy", "entry_section_is_wx",
    "avg_virtual_to_raw_ratio", "has_overlay", "overlay_size",
    "overlay_entropy", "has_known_packer_section", "num_unnamed_sections",
    "entry_section_name_is_standard", "section_name_entropy_mean",
    "is_likely_packed", "ep_outside_sections",
    "num_imported_dlls", "num_imported_functions", "has_suspicious_dlls",
    "num_ordinal_only_imports",
    "num_exports", "has_exports_not_dll",
    "num_resources", "total_resource_size",
    "resource_entropy_mean", "resource_entropy_max",
    "has_version_info", "num_resource_languages",
    "num_strings", "avg_string_length",
    "num_url_strings", "num_ip_strings", "num_registry_strings",
    "num_filepath_strings", "num_base64_strings", "num_api_like_strings",
    "num_wide_strings", "avg_wide_string_length",
    "num_wide_url_strings", "num_wide_ip_strings", "num_wide_registry_strings",
    "num_wide_filepath_strings", "num_wide_base64_strings",
    "num_wide_api_like_strings",
    "has_debug_dir", "has_tls_dir", "has_reloc_dir",
    "has_bound_imports", "has_signature", "has_rich_header",
    "has_delay_load_imports", "has_load_config", "has_exception_dir",
    "is_dotnet", "dotnet_num_metadata_strings",
}
_REQUIRED_NUMERIC_KEYS |= {f"imports_{name}" for name in HIGH_RISK_APIS}

_REQUIRED_STRING_KEYS = {"file_path", "imphash", "dotnet_runtime_version"}
_REQUIRED_LIST_KEYS = {"dotnet_strings"}


class TestExtractorSchema:
    def test_returns_dict(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert isinstance(result, dict)

    def test_all_numeric_keys_present(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        missing = _REQUIRED_NUMERIC_KEYS - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_string_keys_present(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        for k in _REQUIRED_STRING_KEYS:
            assert k in result, f"Missing key: {k}"

    def test_list_keys_present(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        for k in _REQUIRED_LIST_KEYS:
            assert k in result, f"Missing key: {k}"
            assert isinstance(result[k], list), f"{k} should be a list"

    def test_parse_error_key_present(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert "parse_error" in result
        assert result["parse_error"] == 0

    def test_file_path_in_result(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert str(pe32_file) in result["file_path"]


class TestExtractorErrors:
    def test_nonexistent_file_parse_error(self, tmp_path):
        result = PEFeatureExtractor(tmp_path / "ghost.exe").extract()
        assert result["parse_error"] == 1

    def test_nonexistent_file_numeric_features_zero(self, tmp_path):
        result = PEFeatureExtractor(tmp_path / "ghost.exe").extract()
        # All numeric keys should be present and equal 0
        for k in _REQUIRED_NUMERIC_KEYS:
            assert k in result
            assert result[k] == 0, f"{k} should be 0 on parse error"

    def test_non_pe_file_parse_error(self, nonpe_file):
        result = PEFeatureExtractor(nonpe_file).extract()
        assert result["parse_error"] == 1

    def test_truncated_file_parse_error(self, truncated_file):
        result = PEFeatureExtractor(truncated_file).extract()
        assert result["parse_error"] == 1

    def test_non_pe_preserves_file_size(self, nonpe_file):
        result = PEFeatureExtractor(nonpe_file).extract()
        assert result["file_size"] == os.path.getsize(nonpe_file)

    def test_non_pe_preserves_file_entropy(self, nonpe_file):
        result = PEFeatureExtractor(nonpe_file).extract()
        assert result["file_entropy"] == 0.0 or result["file_entropy"] >= 0.0


class TestFileLevel:
    def test_file_size_correct(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["file_size"] == os.path.getsize(pe32_file)

    def test_file_entropy_bounds(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert 0.0 <= result["file_entropy"] <= 8.0

    def test_has_rich_header_set(self, rich_header_file):
        result = PEFeatureExtractor(rich_header_file).extract()
        assert result["has_rich_header"] == 1

    def test_no_rich_header(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["has_rich_header"] == 0


class TestFileHeader:
    def test_is_exe(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["is_exe"] == 1
        assert result["is_dll"] == 0

    def test_is_dll(self, dll_file):
        result = PEFeatureExtractor(dll_file).extract()
        assert result["is_dll"] == 1
        assert result["is_exe"] == 0

    def test_is_32bit(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["is_64bit"] == 0

    def test_is_64bit(self, pe64_file):
        result = PEFeatureExtractor(pe64_file).extract()
        assert result["is_64bit"] == 1

    def test_num_sections_one(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["num_sections"] == 1

    def test_num_sections_two(self, two_section_file):
        result = PEFeatureExtractor(two_section_file).extract()
        assert result["num_sections"] == 2


class TestChecksum:
    def test_checksum_valid(self, valid_checksum_file):
        result = PEFeatureExtractor(valid_checksum_file).extract()
        assert result["checksum_is_valid"] == 1

    def test_checksum_invalid_zero(self, pe32_file):
        # PE built with stored_checksum=0 → checksum_is_valid=0
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["checksum_is_valid"] == 0


class TestSections:
    def test_high_entropy_section_is_likely_packed(self, packed_file):
        result = PEFeatureExtractor(packed_file).extract()
        assert result["is_likely_packed"] == 1

    def test_normal_section_not_packed(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["is_likely_packed"] == 0

    def test_no_overlay(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["has_overlay"] == 0

    def test_section_entropy_bounds(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        for key in ("section_entropy_mean", "section_entropy_max", "section_entropy_min"):
            assert 0.0 <= result[key] <= 8.0, f"{key} out of bounds"


class TestImports:
    def test_no_imports_in_basic_pe(self, pe32_file):
        result = PEFeatureExtractor(pe32_file).extract()
        assert result["num_imported_dlls"] == 0
        assert result["num_imported_functions"] == 0

    def test_suspicious_dll_detected(self, import_pe_file):
        result = PEFeatureExtractor(import_pe_file).extract()
        assert result["has_suspicious_dlls"] == 1

    def test_high_risk_api_detected(self, import_pe_file):
        result = PEFeatureExtractor(import_pe_file).extract()
        assert result["imports_virtualalloc"] == 1

    def test_import_dll_count(self, import_pe_file):
        result = PEFeatureExtractor(import_pe_file).extract()
        assert result["num_imported_dlls"] == 1
        assert result["num_imported_functions"] >= 1


class TestDeterminism:
    def test_deterministic_output(self, pe32_file):
        r1 = PEFeatureExtractor(pe32_file).extract()
        r2 = PEFeatureExtractor(pe32_file).extract()
        # Compare all numeric fields
        for k in _REQUIRED_NUMERIC_KEYS:
            assert r1[k] == r2[k], f"Non-deterministic: {k}"
