# Turnover Prediction

This repository contains the final employee-turnover prediction workflow developed for the project. It trains tabular machine-learning models on HR Excel data, evaluates transfer to a newer holdout source, explains model behavior, and powers the Streamlit dashboard.

## Current final workflow

The final workflow is EDA-driven and centered on the cleaned file1/file2/file3 modeling setup:

1. Impute missing payment values using comparable employees from the better-observed sources.
2. Preprocess raw HR records into clean employee-level modeling tables.
3. Train final models on files 1 and 2 and evaluate generalization on file 3.
4. Generate global and grouped feature-importance reports.
5. Use the final model artifacts in the Streamlit dashboard and what-if simulator.
6. Summarize the full methodology and results in the LaTeX report under `report/`.

## Main commands

```powershell
python imputations.py
python preprocess.py
python train_final_models.py
python scripts/analyze_final_feature_importance.py
python scripts/analyze_grouped_actionable_importance.py
streamlit run app.py
```

The root-level `imputations.py`, `preprocess.py`, and `train_final_models.py` files are compatibility wrappers. The implementation lives in `src/` and `scripts/`.

## Final outputs

- `output/imputation_results.csv`: row-level summary of payment imputations.
- `output/preprocess/`: cleaned modeling tables and preprocessing metadata.
- `output/final/`: model metrics, prediction exports, feature-importance tables, and graphs.
- `artifacts/`: saved `.pkl` model pipelines and metadata.
- `report/final_report.tex`: advisor-facing final report source.
- `report/final_report.pdf`: compiled advisor-facing report.

## Code organization

- `src/imputations.py`: payment-missingness imputation logic.
- `src/preprocess.py`: final EDA-informed preprocessing pipeline.
- `src/final_modeling.py`: final model training and evaluation utilities.
- `src/final_dashboard.py`: helpers that connect final artifacts to the dashboard.
- `src/inference.py`: individual prediction and what-if inference API.
- `scripts/train_final_models.py`: final model training entry point.
- `scripts/analyze_final_feature_importance.py`: global importance and plots.
- `scripts/analyze_grouped_actionable_importance.py`: actionable importance within age and tenure groups.
- `scripts/preprocessing_audit.py`: source-schema and preprocessing audit utility.
- `app.py`: Streamlit dashboard.

## Modeling notes

- Files 1 and 2 are used for model development; file 3 is treated as an external-style generalization test.
- The file-specific aziva coding is normalized conceptually: aziva code 41 in files 1/2 corresponds to aziva code 42 in file 3, and both map to `leave_ind = 1`.
- Non-positive payment records are treated as missing and imputed before modeling.
- Clearly problematic records are removed during preprocessing, except employees above age 75 are retained because they are feasible real employees.
- Models use native class-imbalance handling where appropriate; SMOTE is not part of the final implementation.
- Logistic regression receives the preprocessing/scaling needed for stable estimation.
- Feature importance is reported globally and separately within age/tenure groups to distinguish descriptive context from actionable levers.

## Report

Build the report from the project root with:

```powershell
cd report
pdflatex final_report.tex
pdflatex final_report.tex
```

The second run refreshes references and the table of contents.
