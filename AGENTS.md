# AGENTS.md

Reference notes for future coding agents working in this repository.

## Project summary

This is a Python employee-turnover prediction project. It trains tabular ML models on HR Excel data, produces per-employee turnover probabilities, explains risk drivers, and supports an executive Streamlit dashboard.

The final project workflow is the EDA-informed file1/file2/file3 pipeline. Files 1 and 2 are used for development, and file 3 is used as an external-style generalization test.

## Current entry points

- `imputations.py`: compatibility wrapper for payment imputation.
- `preprocess.py`: compatibility wrapper for final preprocessing.
- `train_final_models.py`: compatibility wrapper for final model training.
- `app.py`: Streamlit executive dashboard using final artifacts and what-if simulation.
- `scripts/train_final_models.py`: final model-training script.
- `scripts/analyze_final_feature_importance.py`: global feature-importance outputs and graphs.
- `scripts/analyze_grouped_actionable_importance.py`: actionable importance within tenure and age groups.
- `scripts/preprocessing_audit.py`: source schema, missingness, and preprocessing audit utility.

## Main commands

```powershell
python imputations.py
python preprocess.py
python train_final_models.py
python scripts/analyze_final_feature_importance.py
python scripts/analyze_grouped_actionable_importance.py
streamlit run app.py
```

To build the final report:

```powershell
cd report
pdflatex final_report.tex
pdflatex final_report.tex
```

## Data and artifacts

- Raw data lives under `data/` and is ignored by git.
- Model artifacts live under `artifacts/` and are mostly generated outputs.
- Modeling outputs live under `output/` and are mostly generated outputs.
- The final LaTeX report lives under `report/` and is intended to be tracked.
- `.env` may exist locally and must not be printed or committed.

## Current core modules

- `src/imputations.py`: payment imputation from comparable employees.
- `src/preprocess.py`: EDA-informed final preprocessing.
- `src/final_modeling.py`: model training, metrics, exports, and artifact writing.
- `src/final_dashboard.py`: dashboard helpers for final model artifacts.
- `src/inference.py`: individual prediction and what-if simulation API.
- `src/datasets.py`: dataset discovery, header handling, and source metadata.
- `src/static_preprocessing.py` and `src/training.py`: older clean/static pipeline support retained for compatibility.

## Modeling notes

- The file-specific aziva coding is normalized conceptually: aziva code 41 in files 1/2 corresponds to aziva code 42 in file 3, and both indicate leaving.
- Payment values that are missing or non-positive are imputed before modeling.
- Clearly invalid records are removed, except age above 75 is retained.
- Native class-imbalance handling is used where appropriate; SMOTE is not part of the final implementation.
- Feature importance should distinguish descriptive features such as age/tenure from actionable levers such as payment, workload, absence, role, and organizational context.

## Worktree notes

- Preserve existing user changes.
- Do not inspect `.env`.
- Avoid changing generated outputs unless the user explicitly asks to rerun or refresh them.
- Use focused commits with an inspected `git status`/`git diff` before staging.
