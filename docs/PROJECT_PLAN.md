# Turnover Prediction Project Plan

## Project Overview

This document outlines the architecture, implementation, and evaluation strategy for the employee turnover prediction system.

## 1. System Architecture

The project uses a modular, layered approach:

```
Data Sources
    ↓
Data Loader & Preprocessor (src/data_loader.py)
    ↓
Feature Engineering (time-series aggregation, derived features)
    ↓
Train/Test Split (70/30 stratified by employee ID)
    ├→ ML Pipeline (Multiple classifiers)
    │   ├→ Best Model Selection (XGBoost - AUC: 0.9724)
    │   ├→ SHAP Feature Importance (src/analysis/shap_explainability.py)
    │   └→ Predictions & Metrics (AUC, F1, Recall@Top20%)
    └→ WizWhy Rule Extraction
        ├→ 1047 if-then rules
        ├→ Rule statistics (IFF Criterion, I.F.)
        └→ Feature frequency analysis
            ↓
        Direct Comparison (threshold=0.50)
            ├→ Metrics Alignment
            ├→ SHAP vs Rule Drivers
            └→ Per-Employee Predictions
```

## 2. Data Architecture

### Input Data
- **Source**: `data/first_file.xlsx` (real HR data)
- **Structure**: Multi-row per employee (time-series observations)
- **Processing**: Aggregated to one row per employee (latest leave_ind as target)

### Train/Test Split Strategy
- **Method**: Stratified 70/30 split by employee ID (seed=42)
- **Outputs**:
  - `split/train_ids.txt`, `split/test_ids.txt` (employee IDs)
  - `split/train_data.xlsx`, `split/test_data.xlsx` (raw data for WizWhy)
- **Benefit**: Same test set used for both ML and WizWhy ensures fair comparison

### Data Processing Pipeline
1. **Loading** (`RealExcelDataLoader.load()`): Read Excel, handle types
2. **Time-Series Aggregation** (`_aggregate_time_series()`): Compress to one row per employee
   - Aggregations: mean, std, trend (linear regression slope) for numeric features
   - Latest value for categorical/flag features
3. **Feature Engineering** (`_feature_engineering()`): Derived features (skewness, ratios, thresholds)
4. **Normalization** (`preprocess()`): StandardScaler for numeric features
5. **Encoding**: One-hot for categorical variables

## 3. Model Implementation

### Classifiers
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Tree-based ensemble (n_estimators=100)
- **XGBoost**: Gradient boosted trees (selected as best model)
  - Validation AUC: 0.9745
  - Test AUC: 0.9724
  - Supports native SHAP computation via pred_contribs
- **AdaBoost**: Adaptive boosting (fallback ensemble component)

### Model Selection Strategy
1. Train all classifiers on training set
2. Evaluate on validation set (random 10% hold-out from training data)
3. Select best model by AUC-ROC
4. Final evaluation on test set (stratified hold-out)

### Ensemble & Voting
- Optional Voting Classifier combines RF, XGB, AdaBoost predictions
- Current best model: XGBoost (univariate performance exceeds ensemble)

## 4. Explainability & Feature Importance

### SHAP (SHapley Additive exPlanations)
- **Purpose**: Model-agnostic feature importance based on coalition game theory
- **Implementation** (`src/analysis/shap_explainability.py`):
  - **XGBoost**: Native SHAP via `model.get_booster().predict(..., pred_contribs=True)` (avoids version incompatibilities)
  - **Other models**: SHAP Explainer wrapper (TreeExplainer, LinearExplainer, PermutationExplainer as appropriate)
  - **Efficiency**: Uses background sample of min(500, len(X)) rows for SHAP calculation
- **Feature aggregation**: Groups correlated time-series features by base name (e.g., `vetek_months_std` + `vetek_months_mean` → `vetek_months`)
- **Aggregation method**: Uses `max` (not sum) to prevent inflating importance of correlated feature pairs
- **Output**: Top-20 features ranked by mean absolute SHAP value

### Feature Name Alignment
- **Challenge**: SHAP uses aggregated feature names (e.g., `vetek_months_std`), WizWhy uses raw names (e.g., `vetek_months`)
- **Solution**: `_base_name()` function strips `_mean`, `_std`, `_trend` suffixes and groups before comparison
- **Readable Names**: `FEATURE_DESCRIPTIONS` maps technical names to human-readable labels

## 5. Evaluation Metrics

### Primary Metrics
- **AUC-ROC**: Area under receiver operating characteristic curve (threshold-independent)
- **F1 Score**: Harmonic mean of precision and recall (threshold=0.50)
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **Recall@Top20%**: Percentage of churners caught in top 20% predicted risk
- **Precision@Top20%**: Percentage of top 20% predictions that are correct churners

### Business Context
- **High Recall@Top20%**: Catch most at-risk employees (minimize missed interventions)
- **High Precision@Top20%**: Prioritize resources on employees most likely to churn
- **WizWhy Characteristics**:
  - Extremely high recall (99.01%) → conservative, sensitive rule design
  - Lower precision (47.86%) → many false positives (flags more employees)
  - Use case: Cast wide net to ensure no one is missed

## 6. Comparison Methodology (ML vs WizWhy)

### Data Alignment
- Both systems evaluate on identical test set (442 employees)
- Threshold for binary classification: 0.50 for both ML probability and WizWhy Concl_Prob
- WizWhy binary predictions parsed directly from Prediction column

### Rule Type Separation (Critical)
- WizWhy rules predict either **Leave** (turnover) or **Stay** (retention)
- Compare_rules.py parses the Then clause to identify rule type
- Feature rankings use only **Leave rules** to isolate actionable churn drivers
- Avoids conflating drivers of loyalty with drivers of churn
- Output shows separate counts: Rule Count (Leave) vs Rule Count (Stay)

### Feature Importance Alignment
- SHAP groups correlated features by base name and uses max aggregation
- WizWhy rule frequency reflects how often a feature appears in decision logic
- Both ranked by their respective importance to turnover prediction only

### Metric Computation
- Same ground truth labels (`leave_ind`) for both
- AUC computed from probability scores
- Binary metrics (F1, Precision, Recall) from 0.50 threshold
- Recall@Top20% and Precision@Top20% computed by ranking on probability scores

### Interpretation
- **ML Model**: High precision, moderate-high recall → discriminative (fewer but more accurate predictions)
- **WizWhy Rules**: Very high recall, moderate precision → sensitive (catches most churners but with false positives)
- **Use Together**: ML for targeted retention efforts, WizWhy for comprehensive monitoring
- **Feature drivers**: SHAP shows global importance via coalition values; WizWhy shows local rule-based logic

## 7. Outputs & Artifacts

### Training Artifacts (Dual Models)
- `artifacts/model_pipeline_random.pkl`: Best model trained on random 60/20/20 split
  - Baseline model for comparison
  - Preserves: model, scaler, feature_names
- `artifacts/model_pipeline_split.pkl`: Best model trained on 70/30 stratified split
  - Used for WizWhy comparison (identical test set)
  - Preserves: model, scaler, feature_names, split_type metadata

### Predictions & Evaluation
- `output/predictions_output.xlsx`: Test set predictions with probabilities and risk categories
- `output/model_comparison.xlsx`:
  - Sheet 1: Metrics comparison (AUC, F1, Precision, Recall, etc. for both models)
  - Sheet 2: Per-employee predictions (probability divergences)
  - Sheet 3: Summary statistics (agreement %, differences)
- `WizWhy/comparison_results.xlsx`:
  - Sheet 1: Metrics comparison (AUC, F1, Precision, Recall, etc.)
  - Sheet 2: SHAP Feature Importance (top 20 risk drivers)
  - Sheet 3: Per-employee predictions (side-by-side comparison with agreement flag)

### Rule Extraction & Analysis
- `WizWhy/if-then.txt`: Raw WizWhy rules (1047 rules, 369KB)
- `WizWhy/rule_feature_comparison.txt`:
  - Feature Comparison: SHAP rank vs WizWhy rule frequency
  - Rule Details: All rules with conditions, probabilities, and record coverage

## 8. Execution Workflow

### Standard Workflow (ML vs WizWhy Comparison)
1. **Initialize Split**:
   ```bash
   python scripts/create_split.py
   ```
   Creates split/ directory with train/test IDs and raw Excel files

2. **WizWhy Analysis** (Manual):
   - Feed `split/train_data.xlsx` to WizWhy as training data
   - Feed `split/test_data.xlsx` to WizWhy for predictions
   - Save predictions as `WizWhy/wizwhy_test_results.txt`

3. **Train ML Models** (both):
   ```bash
   python main.py                      # Random split → model_pipeline_random.pkl
   python main.py --split_file split/  # Stratified split → model_pipeline_split.pkl
   ```

4. **Compare ML Models** (Optional):
   ```bash
   python scripts/compare_models.py
   ```
   Validates that both split strategies yield similar performance

5. **Compare ML vs WizWhy**:
   ```bash
   python scripts/compare_wizwhy.py
   ```
   Uses stratified split model for fair comparison on shared test set
   Computes metrics, SHAP, and per-employee predictions

6. **Rule Analysis**:
   ```bash
   python scripts/compare_rules.py
   ```
   Extracts WizWhy rules and aligns with SHAP feature importance

7. **Visualization** (Optional):
   ```bash
   streamlit run app.py
   ```
   Interactive dashboard for macro/meso/micro views

### Quick Workflow (Baseline Model Only)
For rapid iteration without WizWhy comparison:
```bash
python main.py                 # Trains random split model
streamlit run app.py           # Visualize results
```

## 9. Key Implementation Details

### Path Resolution
- All scripts resolve paths relative to project root
- Uses `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` pattern
- Enables execution from any directory

### Feature Aggregation
- **Time-series features**: Aggregated via mean, std, trend (monthly snapshots)
- **Engineered features**: Derived ratios, flags, thresholds from aggregated values
- **Categorical**: Latest value per employee (not aggregated)

### Handling Imbalance
- **Strategy**: Class weight balancing in XGBoost (scale_pos_weight parameter)
- **Rationale**: ~46% churn rate is relatively balanced, but weight helps with minority class

### Model Persistence
- Artifacts (model, scaler, feature names) pickled to enable:
  - Reproducible SHAP computation without retraining
  - Deployment for inference on new data
  - Explicit feature preservation (prevents silent feature reordering)
