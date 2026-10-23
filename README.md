# PEdigree

PEdigree is a command-line toolkit for training and running a Windows PE malware family classifier. It extracts ~141 static features from PE binaries (headers, sections, imports, strings, entropy, .NET metadata, and more), trains a LightGBM multi-class model to identify malware families, and classifies new samples against that model.

The pipeline has four stages:

```
extract_features.py  →  train.py  →  tune.py (optional)  →  predict.py
```

---

## Project Structure

```
PEdigree/
├── extract_features.py   # Extract PE features to JSONL or CSV
├── train.py              # Train a LightGBM malware family classifier
├── tune.py               # Hyperparameter tuning with Optuna
├── predict.py            # Predict malware family from feature JSON
├── pe_features/          # Core feature extraction library
│   ├── extractor.py      #   PEFeatureExtractor class (~141 features)
│   ├── entropy.py        #   Shannon entropy utilities
│   ├── strings.py        #   ASCII / UTF-16LE string extraction and pattern matching
│   └── suspicious.py     #   Curated lists of suspicious DLLs and high-risk APIs
├── tests/                # pytest test suite (129 tests)
├── docs/                 # Per-script reference documentation
├── models/               # Directory for storing trained model artifacts
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Installation

**Python 3.11 or 3.12** is recommended.

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Extract features

Run `extract_features.py` against a PE file or directory of PE files to produce a feature dataset.

```bash
# Single file → JSONL
python extract_features.py sample.exe -o features.jsonl

# Directory of samples, attach a family label, use 8 workers
python extract_features.py ./samples/mirai/ --label mirai -w 8 -o mirai.jsonl

# Recursive scan, CSV output
python extract_features.py ./corpus/ -r --format csv -o corpus.csv
```

Each output row contains ~141 numeric features plus metadata fields (`file_path`, `parse_error`, `label`). Files that cannot be parsed produce a row with `parse_error=1` and zeroed numeric features.

See [`docs/extract_features.md`](docs/extract_features.md) for the full argument reference and feature schema.

---

### 2. Train a model

Point `train.py` at a directory of per-family NDJSON files (one family per file, filename used as the label):

```
data/
├── mirai.json
├── wannacry.json
└── benign.json
```

```bash
python train.py --data-dir ./data/ --output-dir ./output/
```

This produces a model artifact, a classification report, a confusion matrix, and a feature importance chart in `--output-dir`.

See [`docs/train.md`](docs/train.md) for details on the split strategy, preprocessing, and output artifact format.

---

### 3. Tune hyperparameters (optional)

Run an Optuna search over the same data to find better LightGBM hyperparameters:

```bash
python tune.py --data-dir ./data/ --output-dir ./output/ --n-trials 100
```

This saves `malware_classifier_tuned.pkl` alongside the baseline model. `predict.py` automatically uses the tuned model when it exists.

See [`docs/tune.md`](docs/tune.md) for the search space and output format.

---

### 4. Predict

Classify a PE file by first extracting its features, then running the classifier:

```bash
# Extract features for one file, then predict
python extract_features.py unknown.exe -o /tmp/unknown.jsonl
python predict.py --input /tmp/unknown.jsonl --model ./output/malware_classifier_tuned.pkl
```

Or batch-classify a directory of pre-extracted JSON files:

```bash
python predict.py --input ./feature_files/ --batch --json-output results.json
```

Example output:

```
Sample: unknown.exe
  Predicted family : mirai
  Confidence       : 0.9134 (91.3%)
  Top-5 predictions:
    1. mirai                     0.9134  ####################################
    2. gafgyt                    0.0512  ##
    3. tsunami                   0.0201
    4. benign                    0.0098
    5. wannacry                  0.0055
```

See [`docs/predict.md`](docs/predict.md) for all input modes, arguments, and JSON output format.

---

## Running the Tests

```bash
python -m pytest tests/ -v
```

The test suite covers all four scripts and the `pe_features` library (129 tests). No real PE binaries are required — test fixtures are generated in memory.

---

## Documentation

Detailed reference documentation for each script lives in the [`docs/`](docs/) directory:

- [`docs/extract_features.md`](docs/extract_features.md)
- [`docs/train.md`](docs/train.md)
- [`docs/tune.md`](docs/tune.md)
- [`docs/predict.md`](docs/predict.md)
