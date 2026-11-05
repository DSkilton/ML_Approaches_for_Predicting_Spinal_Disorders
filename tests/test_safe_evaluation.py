"""Tests that guard against the research errors found in the original notebook."""
import numpy as np
import pandas as pd

from safe_evaluation import (
    CLASS_COLUMN, FEATURE_COLUMNS, TARGET, classification_experiment,
    prepare_data, regression_experiment,
)


def fake_data(n=240):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(values, columns=list(FEATURE_COLUMNS))
    frame[TARGET] = (2 * values[:, 0] + 0.5 * values[:, 1] +
                     rng.normal(scale=0.3, size=n))
    frame[CLASS_COLUMN] = np.tile(["normal", "abnormal", "hernia", "spondylolisthesis"], n // 4)
    frame.loc[3, FEATURE_COLUMNS[0]] = np.nan
    return frame


def test_regression_excludes_target_and_returns_finite_scores():
    result = regression_experiment(fake_data())
    assert TARGET not in result["predictors"]
    assert result["n_train"] + result["n_test"] == len(fake_data())
    assert np.isfinite(result["test_r2"])
    assert np.isfinite(result["test_rmse"])


def test_original_labels_remain_unmodified():
    result = classification_experiment(fake_data())
    assert result["labels"] == ["abnormal", "hernia", "normal", "spondylolisthesis"]
    assert TARGET not in result["predictors"]
    assert 0 <= result["test_macro_f1"] <= 1


def test_duplicate_rows_removed_before_split():
    original = fake_data()
    repeated = pd.concat([original, original.iloc[[0]]], ignore_index=True)
    assert len(prepare_data(repeated, TARGET)) == len(original)
