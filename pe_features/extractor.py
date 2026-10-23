"""PE file feature extractor for malware classification."""

import math
import struct
import time
from pathlib import Path

import pefile

from .entropy import shannon_entropy
from .strings import (
    count_patterns,
    count_wide_patterns,
    extract_strings,
    extract_wide_strings,
)
from .suspicious import HIGH_RISK_APIS, SUSPICIOUS_DLLS

# Entropy threshold above which a section is considered "high entropy"
_HIGH_ENTROPY_THRESHOLD = 7.0

# Entropy threshold used in the packed-binary heuristic
_PACKED_ENTROPY_THRESHOLD = 7.2

# PE section characteristic flags
_SCN_MEM_EXECUTE = 0x20000000
_SCN_MEM_WRITE = 0x80000000

# PE data directory indices
_DIR_EXPORT = 0
_DIR_IMPORT = 1
_DIR_RESOURCE = 2
_DIR_EXCEPTION = 3
_DIR_SECURITY = 4
_DIR_RELOC = 5
_DIR_DEBUG = 6
_DIR_TLS = 9
_DIR_LOAD_CONFIG = 10
_DIR_BOUND_IMPORT = 11
_DIR_DELAY_IMPORT = 13
_DIR_COM_DESCRIPTOR = 14  # .NET / CLR runtime header

# DLL characteristic bit flags
_DLLCHAR_DYNAMIC_BASE = 0x0040   # ASLR
_DLLCHAR_NX_COMPAT = 0x0100      # DEP
_DLLCHAR_NO_SEH = 0x0400
_DLLCHAR_GUARD_CF = 0x4000       # Control Flow Guard

# Machine type for 64-bit
_MACHINE_AMD64 = 0x8664

# Rich header marker (sits in the DOS stub)
_RICH_MARKER = b"Rich"

# Known packer section names (normalised to lower, stripped of nulls)
_PACKER_SECTION_NAMES: frozenset[str] = frozenset({
    "upx0", "upx1", "upx2",
    ".mpress1", ".mpress2",
    ".packed",
    ".nsp0", ".nsp1", ".nsp2",
    ".aspack",
    ".adata",
    "petite",
    ".themida",
    ".winlicen",
})

# Standard entry-point section names used by mainstream compilers
_STANDARD_EP_SECTIONS: frozenset[str] = frozenset({".text", "code", ".code"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_present(pe: pefile.PE, idx: int) -> int:
    """Return 1 if data directory *idx* has a non-zero VA and size."""
    try:
        d = pe.OPTIONAL_HEADER.DATA_DIRECTORY[idx]
        return int(d.VirtualAddress != 0 and d.Size != 0)
    except (IndexError, AttributeError):
        return 0


def _compute_checksum(raw: bytes) -> int:
    """Compute the standard PE checksum over *raw* bytes.

    The algorithm: treat the file as an array of 16-bit words (pad with 0 if
    odd length), sum them with 32-bit carry folding, then add the file length.
    The 4 bytes of the stored checksum field are excluded by zeroing them out
    before summation.
    """
    try:
        e_lfanew = struct.unpack_from("<I", raw, 0x3C)[0]
        cksum_offset = e_lfanew + 4 + 20 + 64  # into optional header
    except Exception:
        return 0

    data = bytearray(raw)
    if cksum_offset + 4 > len(data):
        return 0
    data[cksum_offset: cksum_offset + 4] = b"\x00\x00\x00\x00"

    if len(data) % 2:
        data += b"\x00"

    checksum = 0
    for i in range(0, len(data), 2):
        word = struct.unpack_from("<H", data, i)[0]
        checksum += word
        if checksum > 0xFFFFFFFF:
            checksum = (checksum & 0xFFFFFFFF) + 1

    checksum = (checksum & 0xFFFF) + (checksum >> 16)
    checksum += checksum >> 16
    checksum &= 0xFFFF
    checksum += len(raw)
    return checksum & 0xFFFFFFFF


# ECMA-335 metadata root magic ("BSJB")
_METADATA_MAGIC = 0x424A5342


def _parse_dotnet_strings(pe: pefile.PE) -> tuple[bool, str, list[str]]:
    """Parse .NET CLR metadata and return (is_dotnet, runtime_version, strings).

    Walks IMAGE_COR20_HEADER → metadata root → #Strings stream.
    Returns (False, "", []) when the binary is not .NET or parsing fails.
    The strings list contains every unique non-empty entry in the #Strings heap,
    which holds the null-terminated UTF-8 identifiers (type/method/field names,
    namespace names) referenced by the metadata tables.
    """
    # Step 1: confirm the COM descriptor data directory is present
    try:
        com_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[_DIR_COM_DESCRIPTOR]
        if com_dir.VirtualAddress == 0 or com_dir.Size == 0:
            return False, "", []
    except (IndexError, AttributeError):
        return False, "", []

    # Step 2: read IMAGE_COR20_HEADER and locate the metadata root RVA
    # Layout: cb(4) MajorVer(2) MinorVer(2) MetaData.VA(4) MetaData.Size(4) ...
    try:
        clr_hdr = pe.get_data(com_dir.VirtualAddress, max(com_dir.Size, 16))
    except Exception:
        return True, "", []

    if len(clr_hdr) < 16:
        return True, "", []

    metadata_rva, metadata_size = struct.unpack_from("<II", clr_hdr, 8)
    if metadata_rva == 0 or metadata_size == 0:
        return True, "", []

    # Step 3: read and validate the metadata root
    try:
        metadata = pe.get_data(metadata_rva, metadata_size)
    except Exception:
        return True, "", []

    if len(metadata) < 20:
        return True, "", []

    magic, = struct.unpack_from("<I", metadata, 0)
    if magic != _METADATA_MAGIC:
        return True, "", []

    # Step 4: parse the metadata root header to find stream headers
    # Offset 12: VersionLength (DWORD, already padded to 4-byte boundary)
    version_length, = struct.unpack_from("<I", metadata, 12)
    version_padded = (version_length + 3) & ~3

    version_str = ""
    try:
        version_str = (
            metadata[16: 16 + version_length]
            .rstrip(b"\x00")
            .decode("utf-8", errors="replace")
        )
    except Exception:
        pass

    # After the version string: Flags(2) + NumberOfStreams(2)
    header_end = 16 + version_padded
    if header_end + 4 > len(metadata):
        return True, version_str, []

    _flags, num_streams = struct.unpack_from("<HH", metadata, header_end)
    cursor = header_end + 4

    # Step 5: iterate stream headers looking for "#Strings"
    # Each header: Offset(4) Size(4) Name(variable, null-term, 4-byte padded)
    strings_data: bytes | None = None

    for _ in range(num_streams):
        if cursor + 8 > len(metadata):
            break
        stream_off, stream_size = struct.unpack_from("<II", metadata, cursor)
        cursor += 8

        null_pos = metadata.find(b"\x00", cursor)
        if null_pos == -1:
            break
        stream_name = metadata[cursor:null_pos].decode("ascii", errors="replace")

        # Advance past name (including null terminator) to next 4-byte boundary
        name_field_len = (null_pos - cursor + 1 + 3) & ~3
        cursor += name_field_len

        if stream_name == "#Strings":
            end = stream_off + stream_size
            if end <= len(metadata):
                strings_data = metadata[stream_off:end]
            break

    if not strings_data:
        return True, version_str, []

    # Step 6: extract every non-empty null-terminated entry from the heap.
    # Offset 0 is always the empty string; we skip it.
    strings: list[str] = []
    seen: set[str] = set()
    for part in strings_data.split(b"\x00"):
        if not part:
            continue
        try:
            s = part.decode("utf-8", errors="replace")
        except Exception:
            continue
        if s and s not in seen:
            seen.add(s)
            strings.append(s)

    return True, version_str, strings


# ---------------------------------------------------------------------------
# Zero-feature schema
# ---------------------------------------------------------------------------

def _zero_features() -> dict[str, int | float]:
    """Return all numeric features set to zero for schema consistency."""
    keys: tuple[str, ...] = (
        # file-level
        "file_size", "file_entropy", "has_valid_dos_header",
        # file header
        "machine_type", "num_sections", "timestamp",
        "timestamp_is_zero", "timestamp_is_future",
        "file_characteristics", "is_dll", "is_exe", "is_64bit",
        # optional header
        "subsystem", "dll_characteristics",
        "has_aslr", "has_dep", "has_no_seh", "has_cfg",
        "size_of_code", "size_of_initialized_data", "size_of_uninitialized_data",
        "size_of_image", "size_of_headers", "address_of_entry_point",
        "image_base", "linker_major_version", "linker_minor_version",
        "os_major_version", "checksum", "checksum_is_valid", "num_data_directories",
        "ratio_code_to_image",
        # sections
        "num_executable_sections", "num_writable_sections", "num_wx_sections",
        "section_entropy_mean", "section_entropy_max", "section_entropy_min",
        "section_entropy_std", "num_high_entropy_sections",
        "entry_section_entropy", "entry_section_is_wx",
        "avg_virtual_to_raw_ratio", "has_overlay", "overlay_size",
        "overlay_entropy",
        "has_known_packer_section", "num_unnamed_sections",
        "entry_section_name_is_standard", "section_name_entropy_mean",
        "is_likely_packed", "ep_outside_sections",
        # imports
        "num_imported_dlls", "num_imported_functions", "has_suspicious_dlls",
        "num_ordinal_only_imports",
        # exports
        "num_exports", "has_exports_not_dll",
        # resources
        "num_resources", "total_resource_size",
        "resource_entropy_mean", "resource_entropy_max",
        "has_version_info", "num_resource_languages",
        # ascii strings
        "num_strings", "avg_string_length",
        "num_url_strings", "num_ip_strings", "num_registry_strings",
        "num_filepath_strings", "num_base64_strings", "num_api_like_strings",
        # wide (UTF-16LE) strings
        "num_wide_strings", "avg_wide_string_length",
        "num_wide_url_strings", "num_wide_ip_strings", "num_wide_registry_strings",
        "num_wide_filepath_strings", "num_wide_base64_strings",
        "num_wide_api_like_strings",
        # data directories / structural
        "has_debug_dir", "has_tls_dir", "has_reloc_dir",
        "has_bound_imports", "has_signature", "has_rich_header",
        "has_delay_load_imports", "has_load_config", "has_exception_dir",
        # .NET / CLR
        "is_dotnet", "dotnet_num_metadata_strings",
    )
    api_keys = tuple(f"imports_{name}" for name in HIGH_RISK_APIS)
    return {k: 0 for k in keys + api_keys}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PEFeatureExtractor:
    """Extract a flat feature dict from a Windows PE file.

    Usage::

        extractor = PEFeatureExtractor("/path/to/sample.exe")
        features = extractor.extract()

    The returned dict always contains the same keys regardless of whether
    parsing succeeded; ``parse_error`` is set to 1 on failure.
    The ``imphash`` key is a string (MD5 hex) and excluded from the numeric
    zero-features schema.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def extract(self) -> dict[str, int | float | str | list[str]]:
        result: dict[str, int | float | str | list[str]] = {
            "file_path": str(self.file_path),
            "parse_error": 0,
            "imphash": "",
            "dotnet_runtime_version": "",
            "dotnet_strings": [],
        }

        try:
            raw = self.file_path.read_bytes()
        except OSError as exc:
            result["parse_error"] = 1
            result["parse_error_msg"] = str(exc)
            result.update(_zero_features())
            return result

        # File-level features that don't require a valid PE
        result["file_size"] = len(raw)
        result["file_entropy"] = shannon_entropy(raw)
        result["has_rich_header"] = int(_RICH_MARKER in raw[:0x200])

        try:
            pe = pefile.PE(data=raw, fast_load=False)
        except pefile.PEFormatError as exc:
            result["parse_error"] = 1
            result["parse_error_msg"] = str(exc)
            zeros = _zero_features()
            zeros["file_size"] = result["file_size"]
            zeros["file_entropy"] = result["file_entropy"]
            zeros["has_rich_header"] = result["has_rich_header"]
            result.update(zeros)
            return result

        try:
            result.update(self._extract_file_header(pe))
            result.update(self._extract_optional_header(pe, raw))
            result.update(self._extract_sections(pe, raw))
            result.update(self._extract_imports(pe))
            result.update(self._extract_exports(pe, int(result.get("is_dll", 0))))
            result.update(self._extract_resources(pe))
            result.update(self._extract_strings(raw))
            result.update(self._extract_directories(pe))
            result.update(self._extract_dotnet(pe))
            result["has_valid_dos_header"] = int(
                pe.DOS_HEADER.e_magic == 0x5A4D
            )
            try:
                result["imphash"] = pe.get_imphash()
            except Exception:
                result["imphash"] = ""
        except Exception as exc:
            result["parse_error"] = 1
            result["parse_error_msg"] = str(exc)
            for k, v in _zero_features().items():
                result.setdefault(k, v)

        return result

    # ------------------------------------------------------------------
    # Private extraction methods
    # ------------------------------------------------------------------

    def _extract_file_header(self, pe: pefile.PE) -> dict[str, int]:
        fh = pe.FILE_HEADER
        now = int(time.time())
        ts = fh.TimeDateStamp
        chars = fh.Characteristics
        machine = fh.Machine
        return {
            "machine_type": machine,
            "num_sections": fh.NumberOfSections,
            "timestamp": ts,
            "timestamp_is_zero": int(ts == 0),
            "timestamp_is_future": int(ts > now),
            "file_characteristics": chars,
            "is_dll": int(bool(chars & 0x2000)),
            "is_exe": int(not bool(chars & 0x2000) and not bool(chars & 0x1000)),
            "is_64bit": int(machine == _MACHINE_AMD64),
        }

    def _extract_optional_header(
        self, pe: pefile.PE, raw: bytes
    ) -> dict[str, int | float]:
        oh = pe.OPTIONAL_HEADER
        size_image = oh.SizeOfImage or 1
        dll_chars = oh.DllCharacteristics
        stored_cksum = oh.CheckSum

        nonzero_dirs = 0
        try:
            nonzero_dirs = sum(
                1 for d in oh.DATA_DIRECTORY
                if d.VirtualAddress != 0 or d.Size != 0
            )
        except Exception:
            pass

        computed_cksum = _compute_checksum(raw)
        checksum_is_valid = int(
            stored_cksum != 0 and stored_cksum == computed_cksum
        )

        return {
            "subsystem": oh.Subsystem,
            "dll_characteristics": dll_chars,
            "has_aslr": int(bool(dll_chars & _DLLCHAR_DYNAMIC_BASE)),
            "has_dep": int(bool(dll_chars & _DLLCHAR_NX_COMPAT)),
            "has_no_seh": int(bool(dll_chars & _DLLCHAR_NO_SEH)),
            "has_cfg": int(bool(dll_chars & _DLLCHAR_GUARD_CF)),
            "size_of_code": oh.SizeOfCode,
            "size_of_initialized_data": oh.SizeOfInitializedData,
            "size_of_uninitialized_data": oh.SizeOfUninitializedData,
            "size_of_image": oh.SizeOfImage,
            "size_of_headers": oh.SizeOfHeaders,
            "address_of_entry_point": oh.AddressOfEntryPoint,
            "image_base": oh.ImageBase,
            "linker_major_version": oh.MajorLinkerVersion,
            "linker_minor_version": oh.MinorLinkerVersion,
            "os_major_version": oh.MajorOperatingSystemVersion,
            "checksum": stored_cksum,
            "checksum_is_valid": checksum_is_valid,
            "num_data_directories": nonzero_dirs,
            "ratio_code_to_image": oh.SizeOfCode / size_image,
        }

    def _extract_sections(
        self, pe: pefile.PE, raw: bytes
    ) -> dict[str, int | float]:
        sections = pe.sections
        empty: dict[str, int | float] = {
            "num_executable_sections": 0,
            "num_writable_sections": 0,
            "num_wx_sections": 0,
            "section_entropy_mean": 0.0,
            "section_entropy_max": 0.0,
            "section_entropy_min": 0.0,
            "section_entropy_std": 0.0,
            "num_high_entropy_sections": 0,
            "entry_section_entropy": 0.0,
            "entry_section_is_wx": 0,
            "avg_virtual_to_raw_ratio": 0.0,
            "has_overlay": 0,
            "overlay_size": 0,
            "overlay_entropy": 0.0,
            "has_known_packer_section": 0,
            "num_unnamed_sections": 0,
            "entry_section_name_is_standard": 0,
            "section_name_entropy_mean": 0.0,
            "is_likely_packed": 0,
            "ep_outside_sections": 0,
        }
        if not sections:
            return empty

        try:
            ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        except Exception:
            ep = 0

        entropies: list[float] = []
        num_exec = 0
        num_write = 0
        num_wx = 0
        num_high = 0
        vr_ratios: list[float] = []
        entry_entropy = 0.0
        entry_is_wx = 0
        entry_name_is_standard = 0
        has_packer_section = 0
        num_unnamed = 0
        section_name_bytes = bytearray()
        ep_in_section = False

        for section in sections:
            chars = section.Characteristics
            is_exec = bool(chars & _SCN_MEM_EXECUTE)
            is_write = bool(chars & _SCN_MEM_WRITE)

            if is_exec:
                num_exec += 1
            if is_write:
                num_write += 1
            if is_exec and is_write:
                num_wx += 1

            try:
                sec_data = section.get_data()
            except Exception:
                sec_data = b""
            ent = shannon_entropy(sec_data) if sec_data else 0.0
            entropies.append(ent)
            if ent > _HIGH_ENTROPY_THRESHOLD:
                num_high += 1

            raw_size = section.SizeOfRawData
            virt_size = section.Misc_VirtualSize or raw_size
            if raw_size > 0:
                vr_ratios.append(virt_size / raw_size)

            # Section name handling
            try:
                name_bytes = section.Name
                name_str = name_bytes.rstrip(b"\x00").decode("ascii", errors="replace")
            except Exception:
                name_str = ""

            name_lower = name_str.strip().lower()
            section_name_bytes.extend(name_str.encode("ascii", errors="replace"))

            if not name_lower:
                num_unnamed += 1
            if name_lower in _PACKER_SECTION_NAMES:
                has_packer_section = 1

            # Check if entry point falls in this section
            sec_start = section.VirtualAddress
            sec_end = sec_start + (virt_size or raw_size)
            if sec_start <= ep < sec_end:
                entry_entropy = ent
                entry_is_wx = int(is_exec and is_write)
                entry_name_is_standard = int(name_lower in _STANDARD_EP_SECTIONS)
                ep_in_section = True

        mean_ent = sum(entropies) / len(entropies)
        variance = sum((e - mean_ent) ** 2 for e in entropies) / len(entropies)

        # Overlay: bytes after the last section's raw end
        overlay_size = 0
        overlay_entropy = 0.0
        has_overlay = 0
        try:
            last_offset = max(
                s.PointerToRawData + s.SizeOfRawData for s in sections
            )
            if last_offset < len(raw):
                overlay_data = raw[last_offset:]
                overlay_size = len(overlay_data)
                overlay_entropy = shannon_entropy(overlay_data)
                has_overlay = 1
        except Exception:
            pass

        max_ent = max(entropies) if entropies else 0.0
        is_likely_packed = int(max_ent > _PACKED_ENTROPY_THRESHOLD)

        sec_name_entropy = (
            shannon_entropy(bytes(section_name_bytes)) if section_name_bytes else 0.0
        )

        return {
            "num_executable_sections": num_exec,
            "num_writable_sections": num_write,
            "num_wx_sections": num_wx,
            "section_entropy_mean": mean_ent,
            "section_entropy_max": max_ent,
            "section_entropy_min": min(entropies),
            "section_entropy_std": math.sqrt(variance),
            "num_high_entropy_sections": num_high,
            "entry_section_entropy": entry_entropy,
            "entry_section_is_wx": entry_is_wx,
            "avg_virtual_to_raw_ratio": (
                sum(vr_ratios) / len(vr_ratios) if vr_ratios else 0.0
            ),
            "has_overlay": has_overlay,
            "overlay_size": overlay_size,
            "overlay_entropy": overlay_entropy,
            "has_known_packer_section": has_packer_section,
            "num_unnamed_sections": num_unnamed,
            "entry_section_name_is_standard": entry_name_is_standard,
            "section_name_entropy_mean": sec_name_entropy,
            "is_likely_packed": is_likely_packed,
            "ep_outside_sections": int(not ep_in_section and ep != 0),
        }

    def _extract_imports(self, pe: pefile.PE) -> dict[str, int]:
        result: dict[str, int] = {
            "num_imported_dlls": 0,
            "num_imported_functions": 0,
            "has_suspicious_dlls": 0,
            "num_ordinal_only_imports": 0,
        }
        for name in HIGH_RISK_APIS:
            result[f"imports_{name}"] = 0

        if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            return result

        all_api_names: set[str] = set()
        dll_count = 0
        func_count = 0
        ordinal_only_count = 0
        found_suspicious = False

        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            try:
                dll_name = (entry.dll or b"").decode("ascii", errors="replace").lower()
            except Exception:
                dll_name = ""

            dll_count += 1
            if dll_name in SUSPICIOUS_DLLS:
                found_suspicious = True

            for imp in entry.imports:
                func_count += 1
                if imp.name:
                    name_str = imp.name.decode("ascii", errors="replace").lower().strip()
                    all_api_names.add(name_str)
                else:
                    ordinal_only_count += 1

        result["num_imported_dlls"] = dll_count
        result["num_imported_functions"] = func_count
        result["has_suspicious_dlls"] = int(found_suspicious)
        result["num_ordinal_only_imports"] = ordinal_only_count

        for feature_name, api_match in HIGH_RISK_APIS.items():
            if api_match.lower() in all_api_names:
                result[f"imports_{feature_name}"] = 1

        return result

    def _extract_exports(self, pe: pefile.PE, is_dll: int) -> dict[str, int]:
        empty = {"num_exports": 0, "has_exports_not_dll": 0}
        if not hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            return empty
        try:
            num_exports = pe.DIRECTORY_ENTRY_EXPORT.struct.NumberOfFunctions
        except Exception:
            return empty
        return {
            "num_exports": num_exports,
            "has_exports_not_dll": int(num_exports > 0 and not is_dll),
        }

    def _extract_resources(self, pe: pefile.PE) -> dict[str, int | float]:
        empty: dict[str, int | float] = {
            "num_resources": 0,
            "total_resource_size": 0,
            "resource_entropy_mean": 0.0,
            "resource_entropy_max": 0.0,
            "has_version_info": 0,
            "num_resource_languages": 0,
        }
        if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            return empty

        # RT_VERSION type ID
        _RT_VERSION = 16

        num_resources = 0
        total_size = 0
        entropies: list[float] = []
        has_version_info = 0
        languages: set[int] = set()

        try:
            for res_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                type_id = getattr(res_type, "id", None)
                for res_name in res_type.directory.entries:
                    for res_lang in res_name.directory.entries:
                        num_resources += 1
                        lang_id = getattr(res_lang, "id", 0)
                        languages.add(lang_id)
                        if type_id == _RT_VERSION:
                            has_version_info = 1
                        try:
                            data_entry = res_lang.data
                            offset = data_entry.struct.OffsetToData
                            size = data_entry.struct.Size
                            blob = pe.get_data(offset, size)
                            total_size += len(blob)
                            entropies.append(shannon_entropy(blob))
                        except Exception:
                            pass
        except Exception:
            return empty

        if not num_resources:
            return empty

        return {
            "num_resources": num_resources,
            "total_resource_size": total_size,
            "resource_entropy_mean": sum(entropies) / len(entropies) if entropies else 0.0,
            "resource_entropy_max": max(entropies) if entropies else 0.0,
            "has_version_info": has_version_info,
            "num_resource_languages": len(languages),
        }

    def _extract_strings(self, raw: bytes) -> dict[str, int | float]:
        ascii_strings = extract_strings(raw)
        wide_strings = extract_wide_strings(raw)
        result: dict[str, int | float] = {}
        result.update(count_patterns(ascii_strings))
        result.update(count_wide_patterns(wide_strings))
        return result

    def _extract_directories(self, pe: pefile.PE) -> dict[str, int]:
        return {
            "has_debug_dir":          _dir_present(pe, _DIR_DEBUG),
            "has_tls_dir":            _dir_present(pe, _DIR_TLS),
            "has_reloc_dir":          _dir_present(pe, _DIR_RELOC),
            "has_bound_imports":      _dir_present(pe, _DIR_BOUND_IMPORT),
            "has_signature":          _dir_present(pe, _DIR_SECURITY),
            "has_delay_load_imports": _dir_present(pe, _DIR_DELAY_IMPORT),
            "has_load_config":        _dir_present(pe, _DIR_LOAD_CONFIG),
            "has_exception_dir":      _dir_present(pe, _DIR_EXCEPTION),
        }

    def _extract_dotnet(self, pe: pefile.PE) -> dict[str, int | str | list[str]]:
        is_dotnet, runtime_version, strings = _parse_dotnet_strings(pe)
        return {
            "is_dotnet": int(is_dotnet),
            "dotnet_runtime_version": runtime_version,
            "dotnet_num_metadata_strings": len(strings),
            "dotnet_strings": strings,
        }
