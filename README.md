# Machine-learning approaches for predicting spinal disorders

This repository contains an exploratory research notebook (`main.ipynb`) and a
separate, **leakage-safe reference evaluation** (`safe_evaluation.py`).

## Important correction (5 November 2026)

The historical notebook's Random Forest regression experiment accidentally used
`degree_spondylolisthesis` as **both a predictor and the target**. Its reported
regression R² values are therefore **not valid evidence of predictive performance**.
Some other notebook experiments imputed or oversampled before splitting or cross-
validation. Results in the notebook remain for historical traceability only; they
have **not been independently rerun or validated**.

The new evaluation:
1. Selects five commonly available biomechanical predictors, excluding the target.
2. Splits the data into a held-out 20% test partition and an 80% training partition.
3. Fits median imputation inside each training fold and on training data only.
4. Applies SMOTE only to classification training folds via `imblearn.Pipeline`.
5. Uses the dataset's recorded `class` labels without threshold-derived diagnoses.
6. Calculates five-fold training CV and held-out test metrics separately.

**The four recorded classes have mixed clinical meaning.** In particular,
`abnormal` is not a definitive diagnosis. The merged dataset's original
provenance and external validity need further investigation before clinical
claims are made.

## Run

Use Python 3.11 or newer in a fresh virtual environment:

```bash
python -m venv .venv
# Activate .venv using your operating system's command
python -m pip install -r requirements-evaluation.txt
python safe_evaluation.py --data data_set/combined.csv --output results/evaluation.json
python -m pytest -q
```

Outputs in `results/` are locally generated and are intentionally ignored by
Git. **Do not present the notebook's saved outputs as corrected scores.** Rerun
the new evaluation and record package versions, data provenance and a dataset
checksum before interpreting any new numerical findings.

## Historical materials

The original `main.ipynb` is preserved unchanged to retain the research
development history. The previously committed `venv311/` is a local Windows
environment and should not be used as a reproducible dependency source. The
minimal requirements file is for the corrected reference workflow; the original
`requirements.txt` is a historical environment export.
