"""
PE Malware Family Classifier — End-to-end training pipeline.

Usage:
    python train.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Outputs (written to output_dir, default: ./output/):
    malware_classifier.pkl   — trained model + label encoder + feature list
    classification_report.txt
    confusion_matrix.png
    feature_importance.png
"""

import argparse
import glob
import json
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Columns to drop before training
DROP_COLS = ["file_path", "parse_error", "imphash", "label"]

# Categorical columns that need label encoding
CAT_COLS = ["machine_type", "subsystem", "entry_section_name_is_standard"]

# Minimum samples per class to be included in the stratified test split.
# Classes below this threshold are kept in training only.
MIN_SAMPLES_FOR_SPLIT = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir: str) -> pd.DataFrame:
    """Load all NDJSON files in data_dir into one DataFrame."""
    frames = []
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not json_files:
        sys.exit(f"No JSON files found in {data_dir}")

    print(f"Loading {len(json_files)} family files from {data_dir} ...")
    for path in json_files:
        family = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_json(path, lines=True)
        # Use filename as label if 'label' column missing or all null
        if "label" not in df.columns or df["label"].isnull().all():
            df["label"] = family
        else:
            df["label"] = df["label"].fillna(family)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"  Total samples: {len(data):,}")
    print(f"  Families ({data['label'].nunique()}):")
    counts = data["label"].value_counts()
    for fam, n in counts.items():
        print(f"    {fam:<30} {n:>6}")
    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(data: pd.DataFrame):
    """
    Returns (X, y, feature_cols, label_encoders).

    - Drops non-feature columns
    - Label-encodes categorical columns
    - Fills nulls with -1 (sentinel for "field not present in PE")
    """
    y_raw = data["label"].values

    # Drop columns not used as features
    drop = [c for c in DROP_COLS if c in data.columns]
    X = data.drop(columns=drop).copy()

    # Encode known categorical columns
    label_encoders = {}
    for col in CAT_COLS:
        if col not in X.columns:
            continue
        le = LabelEncoder()
        # fillna before astype(str) so NaN → "unknown", not the string "nan"
        X[col] = X[col].fillna("unknown").astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Convert datetime/Timestamp columns to int64 (Unix epoch seconds).
    # pandas 3.x defaults to datetime64[us] rather than datetime64[ns], so
    # compute the divisor from the actual resolution instead of hard-coding 1e9.
    def _dt_to_epoch_seconds(series: pd.Series) -> pd.Series:
        unit = np.datetime_data(series.dtype)[0]
        divisor = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}.get(unit, 10**9)
        return series.astype("int64") // divisor

    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = _dt_to_epoch_seconds(X[col])
        elif X[col].dtype == object:
            # Check if values are Timestamps stored as objects
            sample = X[col].dropna().iloc[0] if X[col].notna().any() else None
            if sample is not None and hasattr(sample, "timestamp"):
                X[col] = _dt_to_epoch_seconds(pd.to_datetime(X[col], errors="coerce"))

    # Fill remaining nulls and label-encode any non-numeric column.
    # pandas 3.x infers pure-string columns as StringDtype rather than object,
    # so check for non-numeric/non-datetime dtype instead of dtype == object.
    for col in X.columns:
        if (not pd.api.types.is_numeric_dtype(X[col])
                and not pd.api.types.is_datetime64_any_dtype(X[col])):
            X[col] = X[col].fillna("unknown")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            X[col] = X[col].fillna(-1)

    feature_cols = list(X.columns)
    print(f"\n  Feature count: {len(feature_cols)}")
    return X.values, y_raw, feature_cols, label_encoders


# ---------------------------------------------------------------------------
# Train / test split with tiny-class handling
# ---------------------------------------------------------------------------
def make_splits(X, y, label_encoder):
    """
    Stratified 70/15/15 split.  Classes with fewer than MIN_SAMPLES_FOR_SPLIT
    samples are placed entirely in the training set (cannot stratify them).
    """
    counts = pd.Series(y).value_counts()
    rare_classes = set(counts[counts < MIN_SAMPLES_FOR_SPLIT].index)

    mask_rare = np.array([yi in rare_classes for yi in y])
    X_rare, y_rare = X[mask_rare], y[mask_rare]
    X_main, y_main = X[~mask_rare], y[~mask_rare]

    if len(rare_classes):
        print(f"\n  Rare families placed entirely in train (< {MIN_SAMPLES_FOR_SPLIT} samples):")
        for rc in sorted(rare_classes):
            print(f"    {rc}  ({counts[rc]} samples)")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_main, y_main, test_size=0.15, stratify=y_main, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, stratify=y_tv, random_state=42
        # 0.176 of 0.85 ≈ 0.15 of total
    )

    # Append rare samples to training set
    if len(X_rare):
        X_train = np.vstack([X_train, X_rare])
        y_train = np.concatenate([y_train, y_rare])

    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    print(f"\n  Split sizes — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val):
    """Train a LightGBM classifier with early stopping."""
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    print("\nTraining LightGBM (early stopping on val macro-F1) ...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test_enc, y_test_raw, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    # Only report on classes present in the test set
    test_labels = sorted(set(y_test_enc) | set(y_pred))
    test_class_names = label_encoder.inverse_transform(test_labels)

    macro_f1 = f1_score(y_test_enc, y_pred, average="macro")
    print(f"\n  Test macro-F1: {macro_f1:.4f}")

    report = classification_report(
        y_test_enc, y_pred,
        labels=test_labels, target_names=test_class_names, digits=3
    )
    print(report)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)
    print(f"  Saved: {report_path}")

    # Confusion matrix (normalised by true class)
    cm = confusion_matrix(y_test_enc, y_pred, labels=test_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = np.where(row_sums == 0, 0.0, cm.astype(float) / safe_sums)
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=test_class_names,
        yticklabels=test_class_names,
        cmap="Blues",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {cm_path}")

    return macro_f1


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_cols, output_dir, top_n=30):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(np.array(feature_cols)[idx], importances[idx], color="steelblue")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fi_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PE malware family classifier")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory with NDJSON family files")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("PE Malware Family Classifier — Training Pipeline")
    print("=" * 60)

    # 1. Load
    data = load_data(args.data_dir)

    # 2. Preprocess
    X, y_raw, feature_cols, cat_encoders = preprocess(data)

    # 3. Encode class labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw)

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test_enc, y_test_raw = make_splits(
        X, y_raw, label_encoder
    )

    # 5. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 6. Evaluate
    macro_f1 = evaluate(model, X_test, y_test_enc, y_test_raw, label_encoder, args.output_dir)

    # 7. Feature importance
    plot_feature_importance(model, feature_cols, args.output_dir)

    # 8. Save model artifact
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "cat_encoders": cat_encoders,
        "macro_f1": macro_f1,
    }
    model_path = os.path.join(args.output_dir, "malware_classifier.pkl")
    joblib.dump(artifact, model_path)
    print(f"  Saved model: {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
