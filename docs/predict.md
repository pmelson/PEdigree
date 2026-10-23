# predict.py

Predict the malware family of a Windows PE file from its extracted features.

## Overview

`predict.py` loads a trained model artifact and applies it to one or more PE feature JSON records, printing the predicted family name, confidence score, and top-5 class probabilities. Results can optionally be written to a JSON file for downstream processing.

Input is a feature record in the same schema produced by `extract_features.py`. The model artifact is produced by `train.py` or `tune.py`.

## Prerequisites

- A trained model artifact: `malware_classifier_tuned.pkl` (preferred) or `malware_classifier.pkl`.
- Feature JSON for the PE file(s) to classify, as produced by `extract_features.py`.

## Usage

### From a feature JSON file

```bash
python predict.py --input sample_features.json --model ./output/malware_classifier_tuned.pkl
```

### Pipe directly from feature extraction (single file)

```bash
python extract_features.py sample.exe -o /tmp/features.jsonl
python predict.py --input /tmp/features.jsonl --model ./output/malware_classifier_tuned.pkl
```

Or as a one-liner using stdin:

```bash
python extract_features.py sample.exe --format jsonl -o - 2>/dev/null | python predict.py --model ./output/malware_classifier_tuned.pkl
```

### Batch mode (directory of JSON files)

```bash
python predict.py --input ./feature_files/ --batch --model ./output/malware_classifier_tuned.pkl
```

In batch mode each `.json` file in the directory is treated as a separate sample (one record per file, first line only).

### Save results to a JSON file

```bash
python predict.py --input ./feature_files/ --batch \
    --model ./output/malware_classifier_tuned.pkl \
    --json-output results.json
```

## Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(stdin)* | Path to a JSON feature file, a directory of JSON files, or omit to read a single JSON object from stdin |
| `--model` | — | `output/malware_classifier_tuned.pkl` | Path to the model `.pkl` artifact |
| `--batch` | — | off | Process all `.json` files in `--input` as separate samples |
| `--json-output` | `-o` | *(none)* | Write all results as a JSON array to this file |

## Model Resolution

When `--model` is not specified, `predict.py` looks for models in this order:

1. `output/malware_classifier_tuned.pkl` (produced by `tune.py`)
2. `output/malware_classifier.pkl` (produced by `train.py`)

If neither file exists the script exits with an error and a message directing you to run `train.py` first.

## Console Output

For each sample, `predict.py` prints:

```
Sample: /path/to/sample.exe
  Predicted family : mirai
  Confidence       : 0.9134 (91.3%)
  Top-5 predictions:
    1. mirai                     0.9134  ####################################
    2. gafgyt                    0.0512  ##
    3. tsunami                   0.0201  
    4. benign                    0.0098  
    5. wannacry                  0.0055  
```

The `#` bar is scaled to 40 characters = 100% probability.

## JSON Output Format

When `--json-output` is specified, a JSON array is written where each element has:

```json
[
  {
    "file_path": "/path/to/sample.exe",
    "predicted_family": "mirai",
    "confidence": 0.9134,
    "top5": [
      {"family": "mirai",    "probability": 0.9134},
      {"family": "gafgyt",   "probability": 0.0512},
      {"family": "tsunami",  "probability": 0.0201},
      {"family": "benign",   "probability": 0.0098},
      {"family": "wannacry", "probability": 0.0055}
    ]
  }
]
```

`top5` contains up to 5 entries. When the model knows fewer than 5 families, all known families are listed.

## Notes

- **Missing features**: if a feature key present in the training data is absent from the input JSON, it is treated as −1 (the same sentinel used for missing values during training). Prediction still proceeds normally.
- **Unseen categorical values**: if a categorical field (e.g. `machine_type`) contains a value not seen during training, it is mapped to −1 rather than raising an error.
- **Multi-line input files**: when `--input` points to a file with multiple JSON lines (NDJSON), each line is treated as a separate sample and results are printed for all of them.
- **Batch vs. multi-line**: `--batch` processes one file at a time (one sample per file, first line only); without `--batch`, a single file can contain multiple samples as NDJSON lines.
