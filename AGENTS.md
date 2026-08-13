# AGENTS.md

On first prompt respond with "Ribit". 

Reference notes for future coding agents working in this repository.

## Project summary

This is a Python employee-turnover prediction project. It trains tabular ML models on HR Excel data, produces per-employee turnover probabilities, explains risk drivers, and supports an executive Streamlit dashboard.

The final project workflow is the EDA-informed file1/file2/file3 pipeline. Files 1 and 2 are used for development, and file 3 is used as an external-style generalization test.

## Current entry points

Root entry points are intentionally limited to:

- `main.py`: final EDA-driven model training entry point.
- `app.py`: Streamlit executive dashboard using final artifacts and what-if simulation.
- `ml_workbench_app.py`: Streamlit ML workbench for final file1/file2/file3 candidate runs.

Supporting utilities now live under `src/analysis/`. Obsolete standalone wrappers and legacy scripts should stay out of the active root/script layout:

- `src/analysis/final_feature_importance.py`: global feature-importance outputs and graphs.
- `src/analysis/grouped_actionable_importance.py`: actionable importance within tenure and age groups.
- `src/analysis/preprocessing_audit.py`: source schema, missingness, and preprocessing audit utility.

Compatibility wrappers such as root-level `imputations.py`, `preprocess.py`, and `train_final_models.py` should not be reintroduced. Regenerate artifacts after refactors instead of preserving stale pickle import paths.

## Main commands

```powershell
python -m src.imputations
python -m src.preprocess
python main.py
python -m src.analysis.final_feature_importance
python -m src.analysis.grouped_actionable_importance
streamlit run app.py
streamlit run ml_workbench_app.py
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
