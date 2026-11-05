# ML Approaches for Predicting Spinal Disorders

Exploratory research into spinal biomechanics, imputation, regression and classification. **Not a clinical diagnostic tool.**

## November 2026 correction

The historical main.ipynb included target leakage: the regression target degree_spondylolisthesis also appeared among input features. Some classifier experiments imputed or resampled outside CV, and a separate exploratory section reassigned original class labels using thresholds. **Previously recorded scores are not validated and must not be quoted as corrected results.**

The notebook regression features are now corrected, but the outputs should be rerun. The original data_set/combined.csv labels are preserved; combined_fixed.csv generated from exploratory thresholds is *not* independently confirmed ground truth.

## Independent baseline evaluation

Use Python 3.11+ and a fresh virtual environment:

    python -m venv .venv
    # Activate the .venv for your OS
    python -m pip install -r requirements-evaluation.txt
    python -m unittest discover -s tests -v
    python evaluation.py --data data_set/combined.csv --task both

The evaluation holds out 20% of data; training-only 5-fold CV fits preprocessing and SMOTE within each fold. Regression excludes the response from predictors. The additional six columns with missing source measurements are not used as independent clinical features. Only fresh rerun metrics may be reported.

The repository previously included a Windows environment under venv311/; it should be regenerated locally rather than committed. External cohorts and clinically confirmed labels are needed before clinical claims.
