"""
PE Malware Family Classifier — Inference script.

Accepts a single JSON object (same schema as training data) and outputs
the predicted malware family plus the top-5 class probabilities.

Usage:
    # From a JSON file (one sample per line; only first line used):
    python predict.py --input sample.json

    # Pipe raw JSON string:
    echo '{...}' | python predict.py

    # From a directory of JSON files (batch mode):
    python predict.py --input /path/to/dir/ --batch
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Reuse preprocessing constants from train.py
from train import CAT_COLS, DATA_DIR, DROP_COLS, OUTPUT_DIR

DEFAULT_MODEL = os.path.join(OUTPUT_DIR, "malware_classifier_tuned.pkl")
FALLBACK_MODEL = os.path.join(OUTPUT_DIR, "malware_classifier.pkl")


def load_model(model_path: str):
    if not os.path.exists(model_path):
        # Try fallback
        if model_path == DEFAULT_MODEL and os.path.exists(FALLBACK_MODEL):
            model_path = FALLBACK_MODEL
        else:
            sys.exit(f"Model not found: {model_path}\nRun train.py first.")
    artifact = joblib.load(model_path)
    artifact["_model_path"] = model_path
    return artifact


def _cell_to_str(value) -> str:
    """Convert a single cell value to a string suitable for label encoding.

    Handles scalars, NaN/None, and sequence values (lists, arrays) that arise
    from columns like dotnet_strings where pd.notna() returns an array rather
    than a scalar bool.
    """
    if isinstance(value, (list, np.ndarray)):
        return str(value)
    try:
        if pd.isna(value):
            return "unknown"
    except (TypeError, ValueError):
        pass
    return str(value)


def preprocess_sample(sample: dict, artifact: dict) -> np.ndarray:
    """Apply the same preprocessing as train.py to a single sample dict."""
    feature_cols = artifact["feature_cols"]
    cat_encoders = artifact["cat_encoders"]

    # Build a single-row DataFrame aligned to training feature columns
    row = {}
    for col in feature_cols:
        row[col] = sample.get(col, np.nan)

    df = pd.DataFrame([row])

    # Encode categorical columns using saved encoders
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        le = cat_encoders.get(col)
        if le is None:
            df[col] = -1
            continue
        val = _cell_to_str(df[col].iloc[0])
        if val in le.classes_:
            df[col] = le.transform([val])
        else:
            # Unseen category — use -1
            df[col] = -1

    # Encode any other non-numeric columns that were label-encoded during training.
    # pandas 3.x infers pure-string columns as StringDtype rather than object,
    # so check for non-numeric dtype instead of dtype == object.
    for col in df.columns:
        if (not pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_datetime64_any_dtype(df[col])):
            le = cat_encoders.get(col)
            if le is not None:
                val = _cell_to_str(df[col].iloc[0])
                df[col] = le.transform([val]) if val in le.classes_ else -1
            else:
                df[col] = -1
        else:
            df[col] = df[col].fillna(-1)

    return df[feature_cols].values


def predict_sample(sample: dict, artifact: dict) -> dict:
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]

    X = preprocess_sample(sample, artifact)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    top5_idx = np.argsort(proba)[-5:][::-1]
    top5 = [
        {"family": label_encoder.inverse_transform([i])[0], "probability": float(proba[i])}
        for i in top5_idx
    ]

    return {
        "predicted_family": pred_label,
        "confidence": float(proba[pred_idx]),
        "top5": top5,
    }


def load_samples_from_path(path: str):
    """Load one or more JSON samples from a file or directory."""
    if os.path.isdir(path):
        samples = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json"):
                fpath = os.path.join(path, fname)
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append((fname, json.loads(line)))
                            break  # one sample per file in batch dir mode
        return samples
    else:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append((os.path.basename(path), json.loads(line)))
        return samples


def main():
    parser = argparse.ArgumentParser(description="Predict malware family from PE features JSON")
    parser.add_argument("--input", "-i", default=None,
                        help="Path to JSON file or directory. Reads stdin if omitted.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Path to model .pkl (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch", action="store_true",
                        help="Process all JSON files in --input directory")
    parser.add_argument("--json-output", "-o", default=None, metavar="FILE",
                        help="Write results as a JSON array to FILE.")
    args = parser.parse_args()

    artifact = load_model(args.model)
    print(f"Model loaded: {artifact['_model_path']}")
    print(f"  Families: {len(artifact['label_encoder'].classes_)}")
    print(f"  Features: {len(artifact['feature_cols'])}")
    print()

    if args.input:
        samples = load_samples_from_path(args.input)
    else:
        # Read single JSON from stdin
        raw = sys.stdin.read().strip()
        if not raw:
            sys.exit("No input provided. Use --input or pipe JSON to stdin.")
        samples = [("stdin", json.loads(raw))]

    json_records = []
    for source, sample in samples:
        result = predict_sample(sample, artifact)
        file_path = sample.get("file_path", source)
        print(f"Sample: {file_path}")
        print(f"  Predicted family : {result['predicted_family']}")
        print(f"  Confidence       : {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
        print("  Top-5 predictions:")
        for rank, entry in enumerate(result["top5"], 1):
            bar = "#" * int(entry["probability"] * 40)
            print(f"    {rank}. {entry['family']:<25} {entry['probability']:.4f}  {bar}")
        print()
        if args.json_output:
            json_records.append({"file_path": file_path, **result})

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(json_records, f, indent=2)
        print(f"JSON results saved: {args.json_output}")


if __name__ == "__main__":
    main()
