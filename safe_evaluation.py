"""Reproducible, leakage-safe reference evaluation for the spinal dataset.

Run: python safe_evaluation.py --data data_set/combined.csv --output results/evaluation.json

The historical main.ipynb is retained for research provenance; its old scores
must not be presented as corrected results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
TARGET = "degree_spondylolisthesis"
CLASS_COLUMN = "class"
# The five biomechanical predictors available for both constituent datasets.
# We intentionally omit the target and sparsely populated secondary features.
FEATURE_COLUMNS = (
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
)


def prepare_data(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    required = set(FEATURE_COLUMNS) | {target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if target in FEATURE_COLUMNS:
        raise ValueError("Target leakage: target is present in predictors")
    numeric = frame.copy()
    for name in FEATURE_COLUMNS + (target,):
        numeric[name] = pd.to_numeric(numeric[name], errors="coerce")
    # A target must never be imputed from test data or treated as a predictor.
    numeric = numeric.dropna(subset=[target])
    # Drop exact records before splitting so they cannot occur in both folds.
    return numeric.drop_duplicates(subset=list(FEATURE_COLUMNS) + [target, CLASS_COLUMN]
                                   if CLASS_COLUMN in numeric else list(FEATURE_COLUMNS) + [target])


def regression_experiment(frame: pd.DataFrame) -> dict:
    cleaned = prepare_data(frame, TARGET)
    x = cleaned.loc[:, FEATURE_COLUMNS].copy()
    y = cleaned[TARGET].copy()
    assert TARGET not in x.columns

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=SEED
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])
    cv = cross_validate(
        model, x_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        n_jobs=1,
        error_score="raise",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "task": "regression",
        "target": TARGET,
        "predictors": list(FEATURE_COLUMNS),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "cv_train_r2_mean": float(np.mean(cv["test_r2"])),
        "cv_train_r2_std": float(np.std(cv["test_r2"])),
        "cv_train_rmse_mean": float(-np.mean(cv["test_rmse"])),
        "test_r2": float(r2_score(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
    }


def classification_experiment(frame: pd.DataFrame) -> dict:
    if CLASS_COLUMN not in frame.columns:
        raise ValueError("Cannot perform classification without recorded 'class' labels")
    cleaned = prepare_data(frame.dropna(subset=[CLASS_COLUMN]), TARGET)
    x = cleaned.loc[:, FEATURE_COLUMNS].copy()
    y = cleaned[CLASS_COLUMN].astype(str).copy()
    # Never reconstruct labels from biomechanical measurements.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y
    )
    model = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
        ("forest", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])
    cv = cross_validate(
        model, x_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=1,
        error_score="raise",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "task": "classification",
        "label_source": "original 'class' column; not threshold-reconstructed",
        "labels": sorted(y.unique().tolist()),
        "predictors": list(FEATURE_COLUMNS),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "cv_train_macro_f1_mean": float(np.mean(cv["test_macro_f1"])),
        "cv_train_macro_f1_std": float(np.std(cv["test_macro_f1"])),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "test_classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }


def evaluate(frame: pd.DataFrame) -> dict:
    return {
        "protocol": "fixed 80/20 holdout; 5-fold CV on training split only",
        "seed": SEED,
        "source_rows": int(len(frame)),
        "limitations": [
            "Recorded labels have mixed specificity: 'abnormal' is not a specific diagnosis.",
            "Dataset provenance, source overlap and clinical generalisability are unverified.",
            "The historical notebook's reported results were not reproduced by this script.",
            "The two tasks use the five consistently populated measurement columns.",
        ],
        "regression": regression_experiment(frame),
        "classification": classification_experiment(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_set/combined.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    results = evaluate(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved corrected evaluation to {args.output}")


if __name__ == "__main__":
    main()
