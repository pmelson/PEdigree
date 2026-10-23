# extract_features.py

Extract features from Windows PE files and write them to JSONL or CSV format for use in model training.

## Overview

`extract_features.py` walks one or more PE files, runs the full feature extraction pipeline on each, and writes the results as a flat record per file. The output feeds directly into `train.py`.

Processing is parallelized across CPU cores by default. Files that cannot be parsed still produce an output row with `parse_error` set to `1` and all numeric features zeroed — so row counts always match file counts.

## Usage

```bash
# Single file → JSONL (default format)
python extract_features.py sample.exe -o features.jsonl

# Directory of PE files, attach a family label to every row
python extract_features.py ./samples/mirai/ --label mirai -o mirai.jsonl

# Directory, CSV output
python extract_features.py ./samples/ --format csv -o features.csv

# Recursive directory scan, 8 parallel workers
python extract_features.py ./corpus/ -r -w 8 -o corpus.jsonl
```

## Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `input` | — | *(required)* | PE file or directory of PE files to process |
| `--output` | `-o` | `features.jsonl` | Output file path |
| `--format` | — | `jsonl` | Output format: `jsonl` or `csv` |
| `--label` | — | *(none)* | Label string attached to every row (e.g. `mirai`, `benign`) |
| `--recursive` | `-r` | off | Recurse into subdirectories |
| `--workers` | `-w` | CPU count | Number of parallel worker processes |

## Output Format

### JSONL (default)

One JSON object per line, one line per file:

```json
{"file_path": "/data/samples/evil.exe", "parse_error": 0, "label": "mirai", "file_size": 49152, "file_entropy": 6.21, "is_dll": 0, "is_exe": 1, ...}
```

### CSV

A header row followed by one data row per file. Column order matches the JSONL keys from the first processed file.

### Always-present fields

| Field | Type | Description |
|---|---|---|
| `file_path` | string | Absolute path to the source file |
| `parse_error` | int (0/1) | `1` if the file could not be parsed as a PE |
| `label` | string | Present only when `--label` is supplied |
| `imphash` | string | MD5 import hash (empty string on parse error) |
| `dotnet_runtime_version` | string | CLR runtime version string, or empty |
| `dotnet_strings` | list[string] | Metadata strings from the .NET `#Strings` heap |

## Feature Schema

The extractor produces ~141 numeric and categorical features grouped into the following categories:

| Category | Features | Description |
|---|---|---|
| File-level | 3 | File size, Shannon entropy, Rich header presence |
| File header | 9 | Machine type, section count, timestamp, DLL/EXE/64-bit flags |
| Optional header | 19 | Subsystem, DLL characteristics, ASLR/DEP/CFG flags, code sizes, linker version, checksum validity |
| Sections | 20 | Entropy statistics, executable/writable counts, packing heuristic, overlay, packer section names |
| Imports | 4 + 41 | DLL/function counts, suspicious DLL flag, per-API binary flags for 41 high-risk Windows APIs |
| Exports | 2 | Export count, anomalous export flag (exports present in a non-DLL) |
| Resources | 6 | Resource count, size, entropy, version info, language count |
| ASCII strings | 8 | String count, average length, URL/IP/registry/filepath/base64/API-like pattern counts |
| Wide strings | 8 | Same as ASCII strings but for UTF-16LE strings |
| Data directories | 9 | Binary presence flags for debug, TLS, relocations, signature, delay imports, etc. |
| .NET / CLR | 3 | `is_dotnet`, runtime version, metadata string count |

## Error Handling

When a file cannot be opened or is not a valid PE, the output row will have:
- `parse_error` = `1`
- `parse_error_msg` — human-readable reason
- All numeric features set to `0`
- `file_size` and `file_entropy` preserved when the file was readable

Errors are reported as warnings to stderr; processing continues with the remaining files.

## Notes

- Output rows are sorted by `file_path` for deterministic, diff-friendly output regardless of which worker processed which file.
- The `--workers` default is `os.cpu_count()`. Set `-w 1` to disable multiprocessing (useful for debugging or on systems with limited RAM).
- The `--label` flag is intended for single-family batches. When training with mixed-family data, omit `--label` and use per-family output files — `train.py` infers labels from filenames.
