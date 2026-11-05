# ML Approaches for Predicting Spinal Disorders

Exploratory research into spinal biomechanics, imputation, regression and classification. **Not a clinical diagnostic tool.**

## November 2026 correction

The historical main.ipynb included target leakage: the regression target degree_spondylolisthesis also appeared among input features. Some classifier experiments imputed or resampled outside CV, and a separate exploratory section reassigned original class labels using thresholds. **Previously recorded scores are not validated and must not be quoted as corrected results.**

The notebook regression features are now corrected, but the outputs should be rerun. The original data_set/combined.csv labels are preserved in the source. Its 3,100 rows contain only 930 distinct measured feature vectors, with 630 groups carrying overlapping labels (abnormal + hernia or abnormal + spondylolisthesis). This is *hierarchical* supervision, not a mutually exclusive four-class ground truth. The safe reference classification benchmark therefore collapses source labels into **normal vs pathology**. It does not invent diagnoses from numeric thresholds. The former four-class notebook results must not be considered validated. combined_fixed.csv generated from exploratory thresholds is not independently confirmed ground truth.

## Independent baseline evaluation

Use Python 3.11+ and a fresh virtual environment:

    python -m venv .venv
    # Activate the .venv for your OS
    python -m pip install -r requirements-evaluation.txt
    python -m unittest discover -s tests -v
    python evaluation.py --data data_set/combined.csv --task both

The evaluation first removes identical measurement vectors, then holds out 20% of the unique vectors; training-only 5-fold CV fits preprocessing and SMOTE within each fold. Regression excludes the response from predictors. The additional six columns with missing source measurements are not used as independent clinical features. Only fresh rerun metrics may be reported.

The repository previously included a Windows environment under venv311/; it should be regenerated locally rather than committed. External cohorts and clinically confirmed labels are needed before clinical claims.

**Further limitation:** Dataset records may include correlated variants of the same underlying case. Without patient/record identifiers or original dataset provenance, we cannot guarantee patient-level separation. Even deduplicated held-out metrics are exploratory, not clinical generalisation evidence.
