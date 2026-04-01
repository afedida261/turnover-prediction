# Project Walkthrough: Turnover Prediction

Complete usage guide for the employee turnover prediction system, including ML pipeline, WizWhy comparison, and interactive dashboard.

## 📂 Folder Structure

```
turnover-prediction/
├── 📄 main.py                    # 🚀 ML PIPELINE ENTRY POINT
├── 📄 app.py                     # Streamlit interactive dashboard
├── 📄 requirements.txt            # Python dependencies
├── 📄 .gitignore                 # Git ignore rules
│
├── 📁 data/                      # Raw input data
│   └── first_file.xlsx           # HR dataset (multiple records per employee)
│
├── 📁 output/                    # ML pipeline outputs
│   └── predictions_output.xlsx   # Model predictions and probabilities
│
├── 📁 artifacts/                 # Trained models & pipeline
│   └── model_pipeline.pkl        # Best model (XGBoost), scaler, feature names
│
├── 📁 split/                     # Train/test split (stratified 70/30)
│   ├── train_ids.txt             # Employee IDs for training
│   ├── test_ids.txt              # Employee IDs for testing
│   ├── train_data.xlsx           # Raw training data (for WizWhy)
│   └── test_data.xlsx            # Raw test data (for WizWhy)
│
├── 📁 WizWhy/                    # WizWhy rule extraction & comparison
│   ├── if-then.txt               # 1047 if-then rules from WizWhy
│   ├── wizwhy_test_results.txt   # WizWhy predictions on test set
│   ├── comparison_results.xlsx   # Metrics, SHAP, per-employee comparison
│   └── rule_feature_comparison.txt  # Feature rankings and rule alignment
│
├── 📁 notebooks/                 # Jupyter notebooks
│   └── EDA_Turnover.ipynb        # Exploratory data analysis
│
├── 📁 src/                       # Source code (core logic)
│   ├── config.py                 # Feature mappings, constants, descriptions
│   ├── data_loader.py            # Data loading, time-series aggregation, preprocessing
│   ├── evaluator.py              # Evaluation metrics
│   ├── models/
│   │   ├── base_model.py         # Abstract base class
│   │   ├── classifiers.py        # LR, RF, XGB, AdaBoost implementations
│   │   └── nn_model.py           # PyTorch neural network
│   └── analysis/
│       ├── fuzzy_importance.py   # Consensus feature importance across models
│       └── shap_explainability.py # SHAP feature importance (XGBoost + others)
│
├── 📁 scripts/                   # Analysis & comparison utilities
│   ├── create_split.py           # Generate 70/30 split and export for WizWhy
│   ├── compare_wizwhy.py         # Compare ML vs WizWhy predictions
│   └── compare_rules.py          # Extract & align WizWhy rules with SHAP
│
└── 📁 docs/                      # Documentation
    ├── README.md                 # Quick start & overview
    ├── PROJECT_PLAN.md           # Architecture & methodology
    └── walkthrough.md            # This file
```

## 🚀 Quick Start

### Step 1: Create Stratified Train/Test Split
```bash
python scripts/create_split.py
```
**Output**: `split/` folder with employee IDs and raw Excel files

**Stratification ensures**:
- Train churn rate ≈ Test churn rate
- Both sets representative of full population
- Reproducible split (seed=42)

**Print output shows**:
```
Split complete (seed=42, test_size=0.30):
  Train employees : 1234 (70.0%)
    Churn rate    : 46.2%
  Test employees  : 529 (30.0%)
    Churn rate    : 46.1%
```

### Step 2: Run WizWhy (Manual)
1. Open `split/train_data.xlsx` in WizWhy
2. Set target = `leave_ind` (column name exact match)
3. Run rule extraction
4. Export predictions to `WizWhy/wizwhy_test_results.txt`

### Step 3a: Train ML Model (Random Split - Baseline)
```bash
python main.py
```
**What it does**:
- Uses default random 60/20/20 stratified split
- Selects best model (typically XGBoost)
- Saves artifacts: `artifacts/model_pipeline_random.pkl`
- Outputs predictions: `output/predictions_output.xlsx`

### Step 3b: Train ML Model (Stratified Split - For WizWhy Comparison)
```bash
python main.py --split_file split/
```
**What it does**:
- Loads `split/train_ids.txt` and `split/test_ids.txt`
- Trains on train employees, evaluates on test employees (shared with WizWhy)
- Selects best model (XGBoost: AUC 0.9724)
- Saves artifacts: `artifacts/model_pipeline_split.pkl`
- Outputs predictions: `output/predictions_output.xlsx`

**Console shows**:
```
Selected Best Model: XGBoost (Validation AUC: 0.9745)
Test Metrics:
  AUC-ROC: 0.9724
  F1: 0.8814
  Precision: 0.9031
  Recall: 0.8719
  Recall@Top20%: 0.9314
  Precision@Top20%: 0.9245
```

### Step 3c: Compare Both Models (Optional)
```bash
python scripts/compare_models.py
```
**What it does**:
- Loads both `model_pipeline_random.pkl` and `model_pipeline_split.pkl`
- Evaluates both on the stratified test set
- Compares: AUC, F1, Precision, Recall, Recall@Top20%, Precision@Top20%
- Identifies divergences in individual predictions
- Generates detailed comparison report

**Outputs**:
- `output/model_comparison.xlsx` with:
  - Metrics Comparison (side-by-side performance)
  - Per-Employee Predictions (individual predictions and divergences)
  - Summary Statistics (prediction agreement %)

**Typical findings**:
- Both models usually perform similarly on test set
- Divergences typically < 10% of employees
- Stratified split may have slight edge due to test-set-aware optimization

### Step 4: Compare Systems
```bash
python scripts/compare_wizwhy.py
```
**Output**: `WizWhy/comparison_results.xlsx` with 3 sheets:

| Sheet | Content |
|-------|---------|
| Metrics Comparison | AUC, F1, Precision, Recall, Recall@Top20%, Precision@Top20% |
| SHAP Feature Importance | Top 20 risk drivers from ML model |
| Per-Employee Predictions | Individual predictions, probabilities, agreement |

**Interpretation**:
- **ML Model** (AUC=0.9724, Precision=90%): High accuracy, discriminative
- **WizWhy** (AUC=0.7599, Recall=99%): Extremely sensitive, catches almost everyone

### Step 5: Analyze Rules
```bash
python scripts/compare_rules.py
```
**Output**: `WizWhy/rule_feature_comparison.txt`

Shows:
- Feature rankings: SHAP rank vs WizWhy rule frequency
- All 1047 if-then rules with conditions and statistics
- Feature alignment (e.g., Tenure ranks #1 in both systems)

### Step 6: Interactive Dashboard
```bash
streamlit run app.py
```
Opens browser window with real-time visualizations.

## 🔑 Key Files & Usage

### `main.py` - ML Pipeline Entry Point

**Default usage** (random 60/20/20 split):
```bash
python main.py
# Outputs: artifacts/model_pipeline_random.pkl, output/predictions_output.xlsx
```

**With shared split** (70/30 from create_split.py):
```bash
python main.py --split_file split/
# Outputs: artifacts/model_pipeline_split.pkl, output/predictions_output.xlsx
```

**Custom data path**:
```bash
python main.py --data_path data/custom_file.xlsx --output_path output/custom_results.xlsx
```

**Arguments**:
- `--data_path` (default: `data/first_file.xlsx`)
- `--output_path` (default: `output/predictions_output.xlsx`)
- `--split_file` (default: None — use this to apply fixed train/test split from create_split.py)

**Console Output**:
- Model training progress and metrics for each classifier
- Best model selection based on validation AUC
- Feature importance from best model only (top 15 features)
- Prediction confidence statistics on test set
- Risk category distribution

### `src/config.py` - Feature Configuration

Edit this to modify features:
```python
NUMERIC_FEATURES = [...]        # Add/remove numeric columns
CATEGORICAL_FEATURES = [...]    # Add/remove categorical columns
FEATURE_DESCRIPTIONS = {...}    # Human-readable feature names
TARGET_COL = "leave_ind"        # Target variable
ID_COLUMNS = ["fictive2"]       # Employee ID column
```

### `src/data_loader.py` - Data Processing

**Three main operations**:
1. **Load**: Read Excel, validate columns, handle types
2. **Aggregate**: Compress multi-row employees to single rows
   - Numeric: mean, std, trend (slope of linear regression)
   - Categorical: latest value
3. **Engineer**: Create derived features
   - Ratios: `salary_change_ratio = salary_change / salary_avg`
   - Flags: `is_high_tenure = tenure_months >= 36`
   - Skewness, z-scores, thresholds

**To add a feature**:
```python
# In _feature_engineering():
df['new_feature'] = df['base_column'].apply(lambda x: ...)
```

### `src/analysis/shap_explainability.py` - Feature Importance

**What it does**:
- Computes SHAP values (SHapley Additive exPlanations) from trained models
- Returns top-20 features ranked by mean |SHAP| value
- Uses max aggregation for correlated feature groups (prevents inflating importance of similar features)

**Model support**:
- **XGBoost**: Native SHAP via `pred_contribs` (fastest, avoids version issues)
- **Random Forest/AdaBoost**: TreeExplainer wrapper
- **Logistic Regression**: LinearExplainer wrapper
- **Voting Classifier**: PermutationExplainer

**Feature name alignment**:
- SHAP aggregates: `vetek_months_std`, `manager_Code_mean`, `avg_Payment_trend`
- Grouped by base name: `vetek_months`, `manager_Code`, `avg_Payment`
- Uses max (not sum) to avoid double-counting correlated features
- Mapped to readable: "Tenure (Months)", "Manager ID", "Average Salary"

### `scripts/create_split.py` - Train/Test Split Generator

**Creates deterministic 70/30 split**:
```python
RANDOM_STATE = 42          # Seed for reproducibility
TEST_SIZE = 0.30           # 30% test, 70% train
STRATIFY = 'leave_ind'     # Balanced churn rate in both sets
```

**Outputs**:
- `split/train_ids.txt` and `split/test_ids.txt`
- `split/train_data.xlsx` and `split/test_data.xlsx`

**Why this matters**:
- Both ML and WizWhy evaluate on identical test set
- Direct comparison is now valid
- Churn rates match across splits (stratification)

### `scripts/compare_wizwhy.py` - System Comparison

**Requires**:
- `WizWhy/wizwhy_test_results.txt` (from manual WizWhy run)
- `output/predictions_output.xlsx` (from `main.py --split_file split/`)
- `artifacts/model_pipeline.pkl` (trained model)

**Computes**:
- Metrics on shared test set (442 employees)
- SHAP top-20 features
- Per-employee predictions with agreement flag

**Critical implementation detail**:
- **WizWhy prediction**: Parse `Prediction` column directly ("more than 0.50" → 1, "No more than 0.50" → 0)
- **ML prediction**: Apply threshold 0.50 to probability scores
- **Both systems**: Evaluated with same ground truth and threshold

### `scripts/compare_models.py` - Dual-Model Comparison

**Compares**:
- Random split model (`model_pipeline_random.pkl`)
- Stratified split model (`model_pipeline_split.pkl`)

**Requirements**:
- Both models must be trained first:
  - `python main.py` → random model
  - `python main.py --split_file split/` → split model
- Split directory must exist: `split/test_ids.txt`

**What it does**:
- Loads both models and test data
- Generates predictions from each
- Computes: AUC, F1, Precision, Recall, Recall@Top20%, Precision@Top20%
- Analyzes per-employee divergences
- Quantifies prediction agreement

**Output**: `output/model_comparison.xlsx` with 3 sheets
- Metrics Comparison: Side-by-side performance metrics
- Per-Employee Predictions: Individual predictions, probabilities, divergences
- Summary Statistics: Agreement %, mean difference, max difference

**Use case**: Validate that random split and stratified split produce similar models (robustness check)

### `scripts/compare_rules.py` - Rule Extraction & Analysis

**Parses WizWhy's if-then.txt** (1047 rules):
- Extracts conditions: "If feature is value AND ..."
- **Critical**: Parses the Then clause to determine if rule predicts **turnover** (Leave) or **retention** (Stay)
- Prevents conflating drivers of churn with drivers of loyalty
- Statistics: probability, records covered, IFF Criterion, I.F.
- Aggregates by feature: separate rule counts for Leave vs Stay rules

**Aligns with SHAP**:
- Groups SHAP features by base name (e.g., `vetek_months_std` + `vetek_months_mean` → `vetek_months`)
- Uses `SHAP_Max` (not sum) to avoid double-counting correlated features
- Ranks both systems by turnover importance (actionable churn drivers)

**Output**: Text file with detailed breakdown
- Feature rankings: SHAP rank vs WizWhy rank, split by Leave/Stay rules
- Rule details: all 1047 rules with prediction type (Leave/Stay) clearly marked

## 📊 Understanding the Metrics

### AUC-ROC (Area Under Receiver Operating Characteristic)
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Probability that the model ranks a random churner higher than a random non-churner
- **ML AUC**: 0.9724 (excellent discrimination)
- **WizWhy AUC**: 0.7599 (rule-based system, fewer decision points)

### Precision & Recall (at threshold 0.50)
- **Precision**: Of predicted churners, what % actually churned?
  - ML: 90.31% (9 out of 10 "high risk" employees actually leave)
  - WizWhy: 47.86% (rule-based, more false positives)
- **Recall**: Of actual churners, what % did we catch?
  - ML: 87.19% (miss 13% of churners)
  - WizWhy: 99.01% (almost no missed churners)

### Recall@Top20% & Precision@Top20%
- **Recall@Top20%**: If we flag top 20% as at-risk, what % of churners are captured?
  - ML: 93.14%
  - WizWhy: ~95% (similar — both catch most churners in top 20%)
- **Precision@Top20%**: Of top 20% flagged, what % are actual churners?
  - ML: 92.45%
  - WizWhy: varies (depends on rule accuracy in that percentile)

### Use Case Interpretation
- **ML Model**: Use for **targeted retention** (focus resources on 87-93% of churners with high precision)
- **WizWhy Rules**: Use for **comprehensive monitoring** (catch 99% but manage false positive burden)

## 💻 Interactive Dashboard (`app.py`)

**Launch**:
```bash
streamlit run app.py
```

**Three Views**:

### 1. Macro View (Company Level)
- Overall churn rate and trends
- Risk category distribution (High/Medium/Low)
- Top at-risk departments
- Tenure vs churn scatterplot

### 2. Meso View (Department Level)
- Filter by organizational unit (Seif)
- Department-specific metrics
- Top 10 at-risk employees in department
- Department churn rate vs company average

### 3. Micro View (Individual Employee)
- Select employee by ID
- Detailed profile: tenure, salary, role, manager
- Current risk prediction
- **What-If Simulation**:
  - Adjust salary ±10%, 20%, etc.
  - Adjust workload (hours) ±10%, 20%, etc.
  - Adjust sick days ±5%, 10%, etc.
  - See real-time risk score updates

## 🛠️ Common Tasks

### Add a New Feature

**1. Add calculation to** `src/data_loader.py`:
```python
# In _feature_engineering():
df['new_feature'] = df['base_column'].apply(my_calculation)
```

**2. Register in** `src/config.py`:
```python
NUMERIC_FEATURES = [..., 'new_feature']
FEATURE_DESCRIPTIONS = {..., 'new_feature': 'Human-Readable Name'}
```

**3. Retrain both models**:
```bash
python main.py                      # Random split model
python main.py --split_file split/  # Stratified split model
python scripts/compare_models.py     # (Optional) Compare the two
```

### Change Model Parameters

**Edit** `main.py` in the `models` dictionary:
```python
models = {
    'XGBoost': XGBoostModel(
        n_estimators=150,      # Increase from 100
        max_depth=6,           # Decrease from 7
        learning_rate=0.05,    # Decrease from 0.1
    )
}
```

### Use Different Data Source

```bash
python scripts/create_split.py       # Uses data/first_file.xlsx by default
python main.py --split_file split/   # Uses split/train_ids.txt and split/test_ids.txt
```

To change source file, edit `create_split.py`:
```python
DATA_PATH = os.path.join("data", "your_file.xlsx")  # Change this line
```

## 📝 Troubleshooting

### Issue: "WizWhy/wizwhy_test_results.txt not found"
**Solution**: Run WizWhy on `split/test_data.xlsx` and save results to `WizWhy/` folder

### Issue: "split/test_ids.txt not found"
**Solution**: Run `python scripts/create_split.py` first

### Issue: "artifacts/model_pipeline.pkl not found"
**Solution**: Run `python main.py --split_file split/` to retrain

### Issue: SHAP computation is slow
**Solution**: It uses a background sample of min(500, len(X)) rows for efficiency. Reduce this in `src/analysis/shap_explainability.py` if needed

## 📚 Further Reading

- **README.md**: Quick start and project overview
- **PROJECT_PLAN.md**: Detailed architecture and methodology
- **src/config.py**: Feature documentation and mappings
- **src/data_loader.py**: Data processing pipeline explanation


