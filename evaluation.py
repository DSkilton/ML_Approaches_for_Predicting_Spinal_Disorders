"""Independent leakage-safe baselines; historical notebook metrics are unverified."""
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
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "degree_spondylolisthesis"
LABEL = "class"
# Original six observed measurements; extra six CSV columns are all missing.
MEASURED_FEATURES = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
                     "sacral_slope", "pelvic_radius", TARGET]
REGRESSION_FEATURES = [name for name in MEASURED_FEATURES if name != TARGET]


def load_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = set(MEASURED_FEATURES + [LABEL]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def regression_inputs(data: pd.DataFrame):
    valid = data.dropna(subset=[TARGET]).copy()
    X = valid[REGRESSION_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(valid[TARGET], errors="coerce")
    ok = y.notna()
    X, y = X.loc[ok], y.loc[ok]
    assert TARGET not in X.columns and LABEL not in X.columns
    return X, y


def classification_inputs(data: pd.DataFrame):
    # Original labels only. Never derive diagnoses using measurement thresholds.
    valid = data.loc[data[LABEL].notna()].copy()
    X = valid[MEASURED_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = valid[LABEL].astype(str).str.strip()
    return X, y


def evaluate_regression(data: pd.DataFrame, folds: int = 5) -> dict:
    X, y = regression_inputs(data)
    Xt, Xh, yt, yh = train_test_split(X, y, test_size=.2, random_state=SEED)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, Xt, yt, cv=cv, scoring="r2")
    model.fit(Xt, yt)
    pred = model.predict(Xh)
    return {
        "features": REGRESSION_FEATURES, "training_rows": len(Xt), "test_rows": len(Xh),
        "train_cv_r2_mean": float(np.mean(scores)), "train_cv_r2_sd": float(np.std(scores)),
        "test_r2": float(r2_score(yh, pred)),
        "test_mae": float(mean_absolute_error(yh, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(yh, pred))),
    }


def evaluate_classification(data: pd.DataFrame, folds: int = 5) -> dict:
    X, y = classification_inputs(data)
    Xt, Xh, yt, yh = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    # Imputation, scaling and SMOTE are learned on each training fold ONLY.
    model = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
        ("model", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, Xt, yt, cv=cv, scoring="f1_macro")
    model.fit(Xt, yt)
    pred = model.predict(Xh)
    return {
        "features": MEASURED_FEATURES, "labels": sorted(y.unique().tolist()),
        "training_rows": len(Xt), "test_rows": len(Xh),
        "train_cv_macro_f1_mean": float(np.mean(scores)),
        "train_cv_macro_f1_sd": float(np.std(scores)),
        "test_accuracy": float(accuracy_score(yh, pred)),
        "test_macro_f1": float(f1_score(yh, pred, average="macro")),
        "test_per_class": classification_report(yh, pred, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_set/combined.csv"))
    parser.add_argument("--task", choices=["both", "regression", "classification"], default="both")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    data = load_data(args.data)
    results = {}
    if args.task in ("both", "classification"):
        results["classification"] = evaluate_classification(data, args.folds)
    if args.task in ("both", "regression"):
        results["regression"] = evaluate_regression(data, args.folds)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
