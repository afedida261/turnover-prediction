# Turnover Prediction

ML project for predicting employee turnover in organizations using real HR data. Includes automated ML pipeline, WizWhy rule extraction comparison, and SHAP explainability analysis.

## Project Overview

This project implements a comprehensive employee churn prediction system:

- **ML Pipeline**: Multiple model implementations (Random Forest, XGBoost, Logistic Regression) with ensemble voting and dual-model comparison.
- **Shared Train/Test Split**: Deterministic 70/30 stratified split by employee ID to enable direct comparison with WizWhy rule-based predictions.
- **SHAP Explainability**: Feature importance analysis using SHapley Additive exPlanations (using max aggregation for correlated feature groups).
- **WizWhy Integration**: Direct comparison of ML predictions vs rule-based predictions on identical test sets, with intelligent rule extraction that separates turnover vs. retention drivers.
- **Evaluation Metrics**: Business-focused metrics including Recall@Top20%, Precision@Top20%, AUC-ROC, F1, Precision, and Recall.

## Project Structure

```
turnover-prediction/
├── data/                        # Raw HR data
│   └── first_file.xlsx          # Input dataset
├── output/                      # ML pipeline outputs
│   └── predictions_output.xlsx  # Model predictions and probabilities
├── artifacts/                   # Trained models and pipeline
│   └── model_pipeline.pkl       # Serialized best model, scaler, feature names
├── split/                       # Train/test split for comparison
│   ├── train_ids.txt            # Employee IDs in training set
│   ├── test_ids.txt             # Employee IDs in test set
│   ├── train_data.xlsx          # Raw training data for WizWhy
│   └── test_data.xlsx           # Raw test data for WizWhy
├── WizWhy/                      # WizWhy rule extraction outputs
│   ├── if-then.txt              # WizWhy's if-then rules (1047 rules)
│   ├── wizwhy_test_results.txt  # WizWhy predictions on test set
│   ├── comparison_results.xlsx  # Metrics, SHAP, and per-employee comparison
│   └── rule_feature_comparison.txt  # Feature ranking and rule alignment
├── notebooks/                   # Jupyter notebooks
│   └── EDA_Turnover.ipynb       # Exploratory data analysis
├── src/                         # Source code
│   ├── config.py                # Feature mappings and constants
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── evaluator.py             # Evaluation metrics
│   ├── models/                  # Model implementations
│   │   ├── base_model.py
│   │   ├── classifiers.py       # RF, XGB, LR, AdaBoost
│   │   └── nn_model.py
│   └── analysis/
│       ├── fuzzy_importance.py  # Consensus importance across models
│       └── shap_explainability.py # SHAP feature importance
├── scripts/                     # Analysis and comparison scripts
│   ├── create_split.py          # Generate 70/30 split and export for WizWhy
│   ├── compare_wizwhy.py        # Compare ML vs WizWhy on shared test set
│   └── compare_rules.py         # Extract and align WizWhy rules with SHAP
├── main.py                      # ML pipeline entry point
├── app.py                       # Streamlit interactive dashboard
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore rules
└── docs/                        # Documentation
    ├── README.md                # This file
    ├── PROJECT_PLAN.md          # Project architecture
    └── walkthrough.md           # Detailed usage guide
```

## Workflow

### Phase 1: Generate Shared Train/Test Split
```bash
python scripts/create_split.py
```
Outputs:
- `split/train_ids.txt` and `split/test_ids.txt` (employee IDs)
- `split/train_data.xlsx` and `split/test_data.xlsx` (raw data for WizWhy)

**Then manually:** Feed `split/train_data.xlsx` and `split/test_data.xlsx` into WizWhy, run analysis, and save predictions as `WizWhy/wizwhy_test_results.txt`.

### Phase 2a: Train ML Model (Random Split - Baseline)
```bash
python main.py
```
Outputs:
- `output/predictions_output.xlsx` (all predictions)
- `artifacts/model_pipeline_random.pkl` (best model on random 60/20/20 split)

### Phase 2b: Train ML Model (Stratified Split - For WizWhy Comparison)
```bash
python main.py --split_file split/
```
Outputs:
- `output/predictions_output.xlsx` (test set predictions)
- `artifacts/model_pipeline_split.pkl` (best model on 70/30 stratified split)

### Phase 2c: Compare Both Models (Optional)
```bash
python scripts/compare_models.py
```
Outputs:
- `output/model_comparison.xlsx` with sheets:
  - Metrics Comparison (side-by-side performance on stratified test set)
  - Per-Employee Predictions (individual predictions and divergences)
  - Summary Statistics (prediction agreement, probability differences)

### Phase 3: Compare ML vs WizWhy
```bash
python scripts/compare_wizwhy.py
```
Uses the stratified split model (`model_pipeline_split.pkl`) for fair comparison.

Outputs:
- `WizWhy/comparison_results.xlsx` with sheets:
  - Metrics Comparison (AUC, F1, Precision, Recall, Recall@Top20%, Precision@Top20%)
  - SHAP Feature Importance (top 20 risk drivers from ML model)
  - Per-Employee Predictions (individual predictions and agreement)

### Phase 4: Analyze Rules and Alignment
```bash
python scripts/compare_rules.py
```
Outputs:
- `WizWhy/rule_feature_comparison.txt` showing:
  - Feature rankings: SHAP rank vs WizWhy rule frequency
  - All 1047 if-then rules with conditions and statistics

## Interactive Dashboard

Run the Streamlit dashboard for real-time visualization:
```bash
streamlit run app.py
```

Features:
- **Macro View**: Company-wide turnover metrics and risk distributions
- **Meso View**: Department/team-level filtering with top 10 at-risk employees
- **Micro View**: Individual employee predictions with what-if simulation (salary, workload, sick days)
