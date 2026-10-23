# analyze_predictions.py

Triage the output of `predict.py` — classify each prediction as a confident match, an ambiguous/suspicious one, or too low-confidence to mean much, and inspect either a single sample or an entire batch.

## Overview

`analyze_predictions.py` reads the JSON array written by `predict.py --json-output` (or any file in that exact schema — `file_path`, `predicted_family`, `confidence`, `top5`) and derives, for every record, a **margin** (the gap between the #1 and #2 candidates in `top5`) and a **tier**:

- **Good match** — high confidence *and* a clear margin over the runner-up. The model is confident and unambiguous.
- **Suspicious** — moderate confidence, or a close race between the top two candidates. Probably not exactly the predicted family, but similar enough to warrant a closer look.
- **Low-confidence** — confidence too low to be meaningful for any candidate.

It needs no model artifact and no ML dependencies — it only reads the predictions JSON, so it works standalone against any `predict.py` output.

## Usage

### Bulk summary

```bash
python analyze_predictions.py --input unclassified_predictions.json
```

### Single-entry report

```bash
# By file_path
python analyze_predictions.py --input unclassified_predictions.json \
    --id 000639ddcb4e40fba3b4a6d47ddf999a19ef952a55c316f95af9b8cb642107e8

# By position in the array
python analyze_predictions.py --input unclassified_predictions.json --index 42
```

### Export full per-record results

```bash
python analyze_predictions.py --input unclassified_predictions.json \
    --csv-output triage.csv --json-output triage.json
```

### Tune the tier thresholds

```bash
python analyze_predictions.py --input unclassified_predictions.json \
    --good-confidence-threshold 0.65 --good-margin-threshold 0.40 --low-confidence-threshold 0.25
```

### From stdin

```bash
python predict.py --input sample_features.json --json-output - 2>/dev/null | python analyze_predictions.py
```

## Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(stdin)* | Path to a `predict.py --json-output` JSON array |
| `--id` | — | *(none)* | Single-entry mode: look up one record by its `file_path` (exact match, then case-insensitive, then substring with suggestions if ambiguous) |
| `--index` | — | *(none)* | Single-entry mode: look up one record by 0-based position (mutually exclusive with `--id`) |
| `--low-confidence-threshold` | — | `0.30` | Confidence floor below which the tier is always `low_confidence`, regardless of margin |
| `--good-confidence-threshold` | — | `0.60` | Confidence floor for the `good_match` tier |
| `--good-margin-threshold` | — | `0.35` | Minimum #1-vs-#2 probability gap for the `good_match` tier |
| `--csv-output` | — | *(none)* | Bulk mode: write full per-record tier data as CSV to this file |
| `--json-output` | `-o` | *(none)* | Bulk mode: write full per-record tier data as a JSON array to this file |
| `--top-n` | — | `15` | Max rows shown per breakdown table in the bulk console summary |
| `--sort` | — | `input` | Export sort order: `input` (unchanged), `confidence`, or `margin` (ascending — most uncertain first, useful for triage) |

## Tier Logic

For each record: `margin = top5[0].probability - top5[1].probability` (if `top5` has fewer than 2 entries, margin equals the top probability, or 0.0 if `top5` is empty/missing).

```
if confidence < low_confidence_threshold:
    tier = low_confidence
elif confidence >= good_confidence_threshold and margin >= good_margin_threshold:
    tier = good_match
else:
    tier = suspicious
```

**Why confidence *and* margin:** since class probabilities sum to at most 1, `margin >= 2*confidence - 1`. At high confidence a wide margin is nearly guaranteed by that inequality alone, so the margin check mainly does independent work in the moderate-confidence band — exactly where two closely-related families (e.g. two RAT variants) can be genuinely hard to tell apart. A record with high confidence but a close runner-up is demoted from `good_match` to `suspicious` rather than being reported as a clean hit.

**Why these defaults:** with 10+ closely-related classes, even a clear, correct prediction often lands well below the near-100% confidence you'd expect from a binary classifier, so the bars are set accordingly rather than near-certainty. Against a real multiclass run of several thousand samples, these defaults produced a non-degenerate split (roughly two-thirds good match, a bit over a quarter suspicious, and a small low-confidence tail) — a reasonable starting point to retune with `--good-confidence-threshold` / `--good-margin-threshold` / `--low-confidence-threshold` for your own data.

## Console Output

### Single-entry mode

```
Sample: 000639ddcb4e40fba3b4a6d47ddf999a19ef952a55c316f95af9b8cb642107e8
  Predicted family : absoluterat
  Confidence       : 0.6463 (64.6%)
  Margin (#1 - #2) : 0.4965 (49.7 pts over njrat)
  Tier             : GOOD MATCH
  Top-5 predictions:
    1. absoluterat               0.6463  ##########################
    2. njrat                     0.1498  #####
    3. revengerat                0.0509  ##
    4. asyncrat                  0.0325  #
    5. limerat                   0.0235
```

### Bulk mode

```
Loaded 7776 records from unclassified_predictions.json

=== Confidence Tier Summary ===
Thresholds: low-confidence < 0.30  |  good-match >= 0.60 confidence AND >= 0.35 margin

  Tier              Count   Percent
  ----------------------------------
  Good match         5182     66.6%
  Suspicious          2213     28.5%
  Low-confidence        381      4.9%
  ----------------------------------
  Total               7776    100.0%

=== Good Match — by family (top 15) ===
    absoluterat               842   16.3%
    njrat                     701   13.5%
    ...
    ... and 12 more families (see --csv-output for full breakdown)

=== Suspicious — by family (top 15) ===
    ...

=== Low-confidence — by family (top 15) ===
    ...

=== Suspicious — most ambiguous family pairs (#1 vs #2) (top 15) ===
    absoluterat          <-> njrat                    61
    remcos               <-> AsyncRat                 38
    ...
```

Family breakdown tables group case-insensitively (so `"SpyGate"` and a hypothetical `"spygate"` elsewhere aren't double-counted), displaying the most commonly seen original casing as the label.

## Export Formats

**CSV** (`--csv-output`) — flat columns for filtering/sorting in a spreadsheet or pandas:

```
file_path,tier,predicted_family,confidence,margin,second_family,second_probability,top5_json
000639dd...,good_match,absoluterat,0.6463212230,0.4964263575,njrat,0.1498,"[{""family"": ""absoluterat"", ...}]"
```

**JSON** (`--json-output`) — the original record with `margin` and `tier` appended:

```json
[
  {
    "file_path": "000639ddcb4e40fba3b4a6d47ddf999a19ef952a55c316f95af9b8cb642107e8",
    "predicted_family": "absoluterat",
    "confidence": 0.6463212230236073,
    "top5": [ ... ],
    "margin": 0.4964263574680659,
    "tier": "good_match"
  }
]
```

## Notes

- **No feature/model dependency**: this tool only reads a predictions JSON array — it doesn't need the `.pkl` model artifact, `pandas`, or `lightgbm`, and doesn't cross-reference feature-extraction data.
- **Malformed records**: a record missing `top5` or `confidence` doesn't raise an error — it's treated as `confidence = 0.0` / no margin and lands in `low_confidence`.
- **Family name casing**: single-record export (CSV/JSON) preserves the original strings byte-for-byte; only the console bulk-summary breakdown tables normalize casing for grouping.
