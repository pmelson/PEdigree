"""Shared pytest fixtures for PEdigree test suite."""

import json
import os
import struct
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pe_features.suspicious import HIGH_RISK_APIS

# ---------------------------------------------------------------------------
# PE binary builder helpers
# ---------------------------------------------------------------------------

_FILE_ALIGN = 0x200
_SECT_ALIGN = 0x1000


def _pad_to(data: bytes, align: int) -> bytes:
    remainder = len(data) % align
    if remainder:
        data += b"\x00" * (align - remainder)
    return data


def make_minimal_pe(
    *,
    machine: int = 0x014C,           # i386 (PE32)
    characteristics: int = 0x0102,   # EXE | 32BIT_MACHINE
    dll_characteristics: int = 0,
    timestamp: int = 0x5F000000,
    section_name: bytes = b".text\x00\x00\x00",
    section_chars: int = 0x60000020,  # CODE | MEM_EXECUTE | MEM_READ
    section_data: bytes | None = None,
    extra_sections: list | None = None,
    ep_rva: int = 0x1000,
    stored_checksum: int = 0,
    rich_header: bool = False,
    import_data_rva: int = 0,
    import_data_size: int = 0,
) -> bytes:
    """Construct a minimal valid PE binary in memory."""
    is_64bit = (machine == 0x8664)
    opt_magic = 0x020B if is_64bit else 0x010B
    opt_hdr_size = 0xF0 if is_64bit else 0xE0  # 240 or 224 bytes

    sections: list[dict] = []
    if section_data is None:
        section_data = b"\xCC" * 0x10
    sections.append({
        "name": section_name[:8].ljust(8, b"\x00"),
        "chars": section_chars,
        "data": _pad_to(section_data, _FILE_ALIGN),
        "virt_size": len(section_data),
    })
    if extra_sections:
        for es in extra_sections:
            sections.append({
                "name": es["name"][:8].ljust(8, b"\x00"),
                "chars": es["chars"],
                "data": _pad_to(es["data"], _FILE_ALIGN),
                "virt_size": len(es["data"]),
            })

    num_sections = len(sections)
    # Use a larger DOS stub when embedding a Rich header so the marker
    # doesn't collide with optional header bytes that follow it.
    dos_size = 0xC0 if rich_header else 64
    headers_raw = dos_size + 4 + 20 + opt_hdr_size + 40 * num_sections
    size_of_headers = ((headers_raw + _FILE_ALIGN - 1) // _FILE_ALIGN) * _FILE_ALIGN

    # Assign virtual and file offsets to each section
    cur_va = ((size_of_headers + _SECT_ALIGN - 1) // _SECT_ALIGN) * _SECT_ALIGN
    cur_raw = size_of_headers
    for sec in sections:
        sec["va"] = cur_va
        sec["raw_offset"] = cur_raw
        sec["raw_size"] = len(sec["data"])
        cur_va = ((cur_va + sec["virt_size"] + _SECT_ALIGN - 1) // _SECT_ALIGN) * _SECT_ALIGN
        cur_raw += sec["raw_size"]

    size_of_image = cur_va
    total_size = cur_raw

    buf = bytearray(total_size)

    # -- DOS header --
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, dos_size)  # e_lfanew
    if rich_header:
        # Place Rich marker in the DOS stub (after DOS header, before PE sig)
        buf[0x80:0x84] = b"Rich"

    # -- PE signature --
    buf[dos_size : dos_size + 4] = b"PE\x00\x00"

    # -- COFF header (20 bytes) --
    coff_off = dos_size + 4
    struct.pack_into(
        "<HHIIIHH", buf, coff_off,
        machine, num_sections, timestamp, 0, 0,
        opt_hdr_size, characteristics,
    )

    # -- Optional header --
    opt_off = coff_off + 20

    if not is_64bit:
        # Standard fields: Magic(H) MajLin(B) MinLin(B) SizeOfCode(I) InitData(I)
        # UninitData(I) EP(I) BaseOfCode(I) BaseOfData(I) = 28 bytes, format HBBIIIIIII
        struct.pack_into(
            "<HBBIIIIII", buf, opt_off,
            opt_magic, 14, 0,
            sections[0]["raw_size"], 0, 0,
            ep_rva, sections[0]["va"], sections[0]["va"],
        )
        # Windows-specific fields (68 bytes):
        # ImageBase(I) SectAlign(I) FileAlign(I)                    = 3I
        # MajOS(H) MinOS(H) MajImg(H) MinImg(H) MajSubs(H) MinSubs(H) = 6H
        # Win32Ver(I) SizeOfImage(I) SizeOfHeaders(I) CheckSum(I)   = 4I
        # Subsystem(H) DllChars(H)                                  = 2H
        # StackRsrv(I) StackCmt(I) HeapRsrv(I) HeapCmt(I) LdrFlags(I) NumDirs(I) = 6I
        # Total: 3*4 + 6*2 + 4*4 + 2*2 + 6*4 = 12+12+16+4+24 = 68 bytes ✓
        win_off = opt_off + 28
        struct.pack_into(
            "<IIIHHHHHHIIIIHHiiiiii", buf, win_off,
            0x00400000, _SECT_ALIGN, _FILE_ALIGN,
            6, 0, 0, 0, 6, 0,
            0, size_of_image, size_of_headers, stored_checksum,
            3, dll_characteristics,
            0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
        )
        # Data directory 1 = imports (at win_off + 68 + 1*8 bytes)
        if import_data_rva and import_data_size:
            dd_off = win_off + 68 + 8
            struct.pack_into("<II", buf, dd_off, import_data_rva, import_data_size)
    else:
        # PE32+ standard fields (24 bytes — no BaseOfData):
        # Magic(H) MajLin(B) MinLin(B) SizeOfCode(I) InitData(I) UninitData(I) EP(I) BaseOfCode(I)
        struct.pack_into(
            "<HBBIIIII", buf, opt_off,
            opt_magic, 14, 0,
            sections[0]["raw_size"], 0, 0,
            ep_rva, sections[0]["va"],
        )
        # PE32+ Windows-specific (88 bytes):
        # ImageBase(Q) SectAlign(I) FileAlign(I)                    = Q+2I
        # MajOS(H) MinOS(H) MajImg(H) MinImg(H) MajSubs(H) MinSubs(H) = 6H
        # Win32Ver(I) SizeOfImage(I) SizeOfHeaders(I) CheckSum(I)   = 4I
        # Subsystem(H) DllChars(H)                                  = 2H
        # StackRsrv(Q) StackCmt(Q) HeapRsrv(Q) HeapCmt(Q)          = 4Q
        # LdrFlags(I) NumDirs(I)                                    = 2I
        # Total: 8+4+4+12+4+4+4+4+4+2+2+8+8+8+8+4+4 = 88 bytes ✓
        win_off = opt_off + 24
        struct.pack_into(
            "<QIIHHHHHHIIIIHHQQQQii", buf, win_off,
            0x0000000140000000, _SECT_ALIGN, _FILE_ALIGN,
            6, 0, 0, 0, 6, 0,
            0, size_of_image, size_of_headers, stored_checksum,
            3, dll_characteristics,
            0x100000, 0x1000, 0x100000, 0x1000,
            0, 16,
        )

    # -- Section headers (40 bytes each) --
    shdr_off = opt_off + opt_hdr_size
    for sec in sections:
        struct.pack_into(
            "<8sIIIIIIHHI", buf, shdr_off,
            sec["name"],
            sec["virt_size"], sec["va"],
            sec["raw_size"], sec["raw_offset"],
            0, 0, 0, 0,
            sec["chars"],
        )
        shdr_off += 40

    # -- Section data --
    for sec in sections:
        buf[sec["raw_offset"] : sec["raw_offset"] + sec["raw_size"]] = sec["data"]

    return bytes(buf)


def _compute_checksum(raw: bytes) -> int:
    """Mirror of extractor._compute_checksum for fixture use."""
    e_lfanew = struct.unpack_from("<I", raw, 0x3C)[0]
    cksum_offset = e_lfanew + 4 + 20 + 64
    data = bytearray(raw)
    data[cksum_offset : cksum_offset + 4] = b"\x00\x00\x00\x00"
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


def make_pe_with_valid_checksum(**kwargs) -> bytes:
    raw = make_minimal_pe(**kwargs)
    cksum = _compute_checksum(raw)
    e_lfanew = struct.unpack_from("<I", raw, 0x3C)[0]
    ck_off = e_lfanew + 4 + 20 + 64
    buf = bytearray(raw)
    struct.pack_into("<I", buf, ck_off, cksum)
    return bytes(buf)


def make_import_pe(dll_name: str, func_names: list) -> bytes:
    """Build a PE32 with a minimal import table for one DLL."""
    import_va_base = 0x2000  # VA where .idata lives

    def va(offset):
        return import_va_base + offset

    n = len(func_names)
    int_offset = 40            # after two IMAGE_IMPORT_DESCRIPTORs (20 bytes each)
    int_size = (n + 1) * 4    # INT: n entries + null, each 4 bytes (PE32)
    iat_offset = int_offset + int_size
    iat_size = int_size
    names_offset = iat_offset + iat_size

    # Compute byte offsets for each IMAGE_IMPORT_BY_NAME
    name_offsets = []
    cur = names_offset
    for fn in func_names:
        name_offsets.append(cur)
        cur += 2 + len(fn) + 1  # hint(2) + name + null
    dll_name_offset = cur
    cur += len(dll_name) + 1

    idata = bytearray(cur)

    # IMAGE_IMPORT_DESCRIPTOR (20 bytes): OriginalFirstThunk TimeDateStamp ForwarderChain Name FirstThunk
    struct.pack_into("<IIIII", idata, 0,
        va(int_offset), 0, 0, va(dll_name_offset), va(iat_offset))
    # Null descriptor at offset 20 — already zeros

    # INT and IAT entries
    for i, noff in enumerate(name_offsets):
        struct.pack_into("<I", idata, int_offset + i * 4, va(noff))
        struct.pack_into("<I", idata, iat_offset + i * 4, va(noff))
    # Null terminators already zeros

    # IMAGE_IMPORT_BY_NAME entries
    for noff, fn in zip(name_offsets, func_names):
        struct.pack_into("<H", idata, noff, 0)  # hint
        fn_bytes = fn.encode("ascii") + b"\x00"
        idata[noff + 2 : noff + 2 + len(fn_bytes)] = fn_bytes

    # DLL name string
    dll_bytes = dll_name.encode("ascii") + b"\x00"
    idata[dll_name_offset : dll_name_offset + len(dll_bytes)] = dll_bytes

    return make_minimal_pe(
        section_data=b"\xCC" * 0x10,
        extra_sections=[{
            "name": b".idata\x00\x00",
            "chars": 0xC0000040,
            "data": bytes(idata),
        }],
        import_data_rva=import_va_base,
        import_data_size=len(idata),
    )


# ---------------------------------------------------------------------------
# ML fixture helpers
# ---------------------------------------------------------------------------

def _feature_columns() -> list:
    cols = [
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
        "overlay_entropy",
        "has_known_packer_section", "num_unnamed_sections",
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
    ]
    for name in HIGH_RISK_APIS:
        cols.append(f"imports_{name}")
    return cols


def _make_sample_df(n_per_class: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    families = ["familyA", "familyB", "familyC"]
    cols = _feature_columns()
    rows = []
    for fam in families:
        for _ in range(n_per_class):
            row = {c: float(rng.integers(0, 100)) for c in cols}
            row["machine_type"] = str(rng.choice([332, 34404]))
            row["subsystem"] = str(rng.choice([2, 3]))
            row["entry_section_name_is_standard"] = str(rng.choice([0, 1]))
            row["label"] = fam
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def minimal_pe32_bytes():
    return make_minimal_pe()


@pytest.fixture(scope="session")
def minimal_pe64_bytes():
    return make_minimal_pe(machine=0x8664, characteristics=0x0022)


@pytest.fixture(scope="session")
def dll_pe_bytes():
    return make_minimal_pe(characteristics=0x2022)  # DLL flag set


@pytest.fixture(scope="session")
def two_section_pe_bytes():
    return make_minimal_pe(
        extra_sections=[{
            "name": b".data\x00\x00\x00",
            "chars": 0xC0000040,
            "data": b"\x00" * 0x10,
        }]
    )


@pytest.fixture(scope="session")
def packed_pe_bytes():
    import os as _os
    data = _os.urandom(0x200)  # high-entropy data → is_likely_packed=1
    return make_minimal_pe(section_data=data, section_chars=0x60000020)


@pytest.fixture(scope="session")
def rich_header_pe_bytes():
    return make_minimal_pe(rich_header=True)


@pytest.fixture(scope="session")
def valid_checksum_pe_bytes():
    return make_pe_with_valid_checksum()


@pytest.fixture(scope="session")
def import_pe_bytes():
    return make_import_pe("ws2_32.dll", ["VirtualAlloc"])


@pytest.fixture
def pe32_file(tmp_path, minimal_pe32_bytes):
    p = tmp_path / "sample.exe"
    p.write_bytes(minimal_pe32_bytes)
    return p


@pytest.fixture
def pe64_file(tmp_path, minimal_pe64_bytes):
    p = tmp_path / "sample64.exe"
    p.write_bytes(minimal_pe64_bytes)
    return p


@pytest.fixture
def dll_file(tmp_path, dll_pe_bytes):
    p = tmp_path / "sample.dll"
    p.write_bytes(dll_pe_bytes)
    return p


@pytest.fixture
def two_section_file(tmp_path, two_section_pe_bytes):
    p = tmp_path / "two_sections.exe"
    p.write_bytes(two_section_pe_bytes)
    return p


@pytest.fixture
def packed_file(tmp_path, packed_pe_bytes):
    p = tmp_path / "packed.exe"
    p.write_bytes(packed_pe_bytes)
    return p


@pytest.fixture
def rich_header_file(tmp_path, rich_header_pe_bytes):
    p = tmp_path / "rich.exe"
    p.write_bytes(rich_header_pe_bytes)
    return p


@pytest.fixture
def valid_checksum_file(tmp_path, valid_checksum_pe_bytes):
    p = tmp_path / "valid_cksum.exe"
    p.write_bytes(valid_checksum_pe_bytes)
    return p


@pytest.fixture
def import_pe_file(tmp_path, import_pe_bytes):
    p = tmp_path / "imports.exe"
    p.write_bytes(import_pe_bytes)
    return p


@pytest.fixture
def nonpe_file(tmp_path):
    p = tmp_path / "notape.bin"
    p.write_bytes(b"\x00\x01\x02\x03" * 64)
    return p


@pytest.fixture
def truncated_file(tmp_path):
    p = tmp_path / "truncated.exe"
    p.write_bytes(b"MZ" + b"\x00" * 18)
    return p


@pytest.fixture
def sample_df():
    return _make_sample_df()


@pytest.fixture
def trained_artifact(tmp_path, sample_df):
    """Train a tiny LightGBM model and return (artifact_dict, pkl_path)."""
    from train import make_splits, preprocess

    X, y_raw, feature_cols, cat_encoders = preprocess(sample_df.copy())
    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw)
    X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, _ = make_splits(
        X, y_raw, label_encoder
    )
    model = lgb.LGBMClassifier(
        n_estimators=10, learning_rate=0.1, num_leaves=7,
        class_weight="balanced", random_state=42, verbose=-1, n_jobs=1,
    )
    model.fit(X_train, y_train_enc)
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "cat_encoders": cat_encoders,
        "macro_f1": 0.5,
    }
    pkl_path = tmp_path / "malware_classifier.pkl"
    joblib.dump(artifact, pkl_path)
    return artifact, str(pkl_path)


@pytest.fixture
def sample_feature_row(sample_df):
    return sample_df.iloc[0].drop("label").to_dict()


@pytest.fixture
def ndjson_data_dir(tmp_path, sample_df):
    for family, group in sample_df.groupby("label"):
        path = tmp_path / f"{family}.json"
        with path.open("w") as f:
            for _, row in group.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
    return tmp_path
