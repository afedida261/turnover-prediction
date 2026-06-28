# Project Plan

## Objective

The project predicts employee turnover (`leave_ind`) from HR records and translates model output into explanations that can support advisor review, HR analysis, and dashboard-based exploration.

## Final methodology

The final modeling design separates three concerns:

1. Data repair and cleaning.
2. Predictive modeling and external-style validation.
3. Explainability and actionability analysis.

Files 1 and 2 are used as the primary development data. File 3 is held out as the newest/generalization source. This split is intentionally stricter than a random split because it tests whether patterns learned from earlier sources transfer to a different file.

## Data preparation

The preprocessing pipeline handles the major EDA findings:

- Payment missingness and non-positive payment values are repaired using an imputation model based on comparable employees from files with more reliable payment observations.
- Non-positive workload, negative illness, underage employees, and other clearly problematic records are removed.
- Employees above age 75 are retained because they are plausible real cases.
- Leakage-prone departure fields, time identifiers, and direct target proxies are excluded from modeling.
- Categorical fields are encoded inside model pipelines, and numeric fields are imputed/scaled in train-fitted preprocessing steps.

## Modeling

The final training script compares the project model family using EDA-informed preprocessing. The implementation emphasizes interpretable tabular baselines and tree ensembles rather than synthetic resampling. Native class-imbalance controls are used where supported by the model.

Primary outputs are written to `output/final/`, and fitted pipelines are written to `artifacts/`.

## Explainability

Feature importance is produced in three complementary ways:

- model-native or SHAP-style global importance for overall ranking;
- permutation-style reliance on the held-out file 3 data;
- grouped actionable importance within age and tenure bands.

The grouped analysis is important because age and tenure are strong descriptive predictors but are not direct management levers. The grouped reports help identify which compensation, workload, absence, role, or organizational features matter within comparable employee contexts.

## Dashboard integration

The Streamlit app keeps the existing executive-dashboard appearance while loading selectable final/workbench artifacts. The what-if simulator now focuses on levers supported by the final importance analysis, so simulated changes better match the model's learned behavior.

## Final deliverables

- Cleaned and organized source code in `src/`, with active analysis utilities under `src/analysis/` and obsolete standalone wrappers removed from the active tree.
- Final model artifacts in `artifacts/`.
- Evaluation and explainability outputs in `output/final/`.
- Updated Streamlit dashboard in `app.py`.
- Configurable ML workbench in `ml_workbench_app.py`.
- Advisor-facing LaTeX report in `report/final_report.tex` and `report/final_report.pdf`.
