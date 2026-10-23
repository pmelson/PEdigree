#!/usr/bin/env python3
"""CLI for extracting PE file features for malware classification training.

Usage examples::

    # Single file, JSONL output
    python extract_features.py sample.exe -o features.jsonl

    # Directory, CSV output, with a family label
    python extract_features.py ./samples/ --label mirai --format csv -o mirai.csv

    # Recursive directory scan, 8 parallel workers
    python extract_features.py ./corpus/ -r -w 8 -o corpus.jsonl
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pe_features import PEFeatureExtractor


def _collect_files(input_path: Path, recursive: bool) -> list[Path]:
    """Return a sorted list of file paths to process."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        print(f"Error: '{input_path}' is not a file or directory.", file=sys.stderr)
        sys.exit(1)

    if recursive:
        return sorted(p for p in input_path.rglob("*") if p.is_file())
    return sorted(p for p in input_path.iterdir() if p.is_file())


def _extract_one(args: tuple[str, str | None]) -> dict:
    """Worker function: extract features from a single file.

    Accepts a tuple so it can be passed through ProcessPoolExecutor.map.

    Args:
        args: (file_path_str, label)

    Returns:
        Feature dict with optional 'label' key.
    """
    file_path_str, label = args
    extractor = PEFeatureExtractor(file_path_str)
    features = extractor.extract()
    if label is not None:
        features["label"] = label
    return features


def _write_jsonl(features_iter, output_path: Path) -> int:
    """Write features as JSON Lines. Returns count of rows written."""
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for features in features_iter:
            fh.write(json.dumps(features, allow_nan=False) + "\n")
            count += 1
    return count


def _write_csv(features_iter, output_path: Path) -> int:
    """Write features as CSV. Returns count of rows written."""
    count = 0
    writer = None
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        for features in features_iter:
            if writer is None:
                fieldnames = list(features.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(features)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PE file features for malware classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="PE file or directory of PE files to process.",
    )
    parser.add_argument(
        "-o", "--output",
        default="features.jsonl",
        help="Output file path (default: features.jsonl).",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format: 'jsonl' (default) or 'csv'.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label string to attach to all rows (e.g. 'mirai', 'benign').",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes (default: cpu count).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    files = _collect_files(input_path, args.recursive)

    if not files:
        print("No files found to process.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Processing {len(files)} file(s) with {args.workers} worker(s) "
        f"-> {output_path} [{args.format}]"
    )

    worker_args = [(str(f), args.label) for f in files]

    # Collect results in submission order using a progress bar
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_extract_one, wa): wa for wa in worker_args}
        with tqdm(total=len(futures), unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    path_str = futures[future][0]
                    print(
                        f"\nWarning: unexpected error processing '{path_str}': {exc}",
                        file=sys.stderr,
                    )
                pbar.update(1)

    # Sort by file_path for deterministic output
    results.sort(key=lambda r: r.get("file_path", ""))

    if args.format == "csv":
        count = _write_csv(iter(results), output_path)
    else:
        count = _write_jsonl(iter(results), output_path)

    errors = sum(1 for r in results if r.get("parse_error", 0))
    print(f"Done. {count} rows written ({errors} parse error(s)).")


if __name__ == "__main__":
    main()
