# Project Walkthrough: Turnover Prediction

This document explains the structure and usage of the Turnover Prediction project. This project is designed to predict employee turnover using various machine learning models (Logistic Regression, Random Forest, XGBoost, etc.) based on HR data.

## 📂 Folder Structure

The project has transitioned to a modular structure using the `src/` directory.

```
turnover-prediction/
├── main.py                 # 🚀 ENTRY POINT: Run this to execute the pipeline
├── EDA_Turnover.ipynb      # Jupyter Notebook for Exploratory Data Analysis
├── PROJECT_PLAN.md         # Architecture and roadmap document
├── factory1.xlsx           # Example data file (Excel)
│
├── src/                    # 🧠 CORE LOGIC (Work here)
│   ├── config.py           # Configuration (Feature names, columns, lists)
│   ├── data_loader.py      # Data loading, Preprocessing & Feature Engineering
│   ├── evaluator.py        # Evaluation metrics (AUC, F1, Recall@Top20%)
│   └── models/             # Model Implementations
│       ├── base_model.py   # Abstract base class
│       ├── classifiers.py  # Main classifiers (LR, RF, XGB, AdaBoost)
│       └── nn_model.py     # Neural Network model
│
└── [Legacy/Duplicate Files] # ⚠️ Avoid editing these, use src/ instead
    ├── classification_models.py (Legacy version of src/models/classifiers.py)
    ├── nn_model.py             (Legacy version of src/models/nn_model.py)
    ├── base_model.py           (Legacy version of src/models/base_model.py)
    └── main_old.py             (Old entry point)
```

## 🔑 Key Files Explained

### 1. `main.py` (The entry point)
This is the script you run to execute the full workflow.
- **What it does**:
  1. Loads data using `src.data_loader.RealExcelDataLoader`.
  2. Preprocesses data (cleaning, feature engineering).
  3. Splits data into Train/Validation/Test.
  4. Trains multiple models (Logistic Regression, Random Forest, XGBoost, etc.).
  5. Evaluates models and selects the best one based on AUC.
  6. Generates an output Excel file with predictions and risk categories.
- **Usage**:
  ```bash
  python main.py --data_path factory1.xlsx --output_path results.xlsx
  ```

### 2. `src/config.py` (The control center)
**Edit this file if you want to add/remove features.**
- Defines `TARGET_COL` (the target variable).
- Lists `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, etc.
- Contains English-to-Hebrew feature descriptions (`FEATURE_DESCRIPTIONS`) for readability.

### 3. `src/data_loader.py` (The engine room)
Handles all data adjustments.
- **Class `RealExcelDataLoader`**:
    - **`_aggregate_time_series`**: Since the data has multiple rows per employee (time-series), this function compresses them into one row per employee (calculating trends, averages, etc.).
    - **`_feature_engineering`**: Creates new features like `salary_skewness`, `tenure_ratio`, `is_young`, etc. **Add new feature logic here.**
    - **`preprocess`**: Orchestrates cleaning, encoding, and normalization.

### 4. `src/models/` (The brains)
- **`classifiers.py`**: Contains standard ML models wrapped in a consistent interface.
- **`nn_model.py`**: A PyTorch Neural Network implementation.

### 5. `src/analysis/` (The insight generator)
- **`fuzzy_importance.py`**: Implements a **Fuzzy Information Fusion** approach.
    - Instead of relying on one model's feature importance (which can be biased), this module aggregates scores from *all* trained models.
    - Uses fuzzy logic (Low/Medium/High membership) to create a robust "Consensus Importance Score".

## 📊 Key Metrics Explained

The project uses specific metrics tailored for high-churn datasets:

1.  **Precision@Top20%**: "Of the 20% employees predicted as highest risk, what percentage *actually* left?"
    - **Goal**: > 80%. (Ensures the "High Risk" list is accurate).
2.  **Recall@Top50%**: "What percentage of *all* churners did we catch in the top half of the population?"
    - **Goal**: > 85%. (Ensures we don't miss many at-risk employees).
    - *Note*: We use Top 50% instead of Top 20% because the churn rate is ~46%, so it's mathematically impossible to catch everyone in the top 20%.

## 🛠️ How to Work with This Folder

### If you want to...

**1. Add a new feature:**
1.  Open `src/data_loader.py` and add the calculation logic in `_feature_engineering`.
2.  Open `src/config.py` and add the new feature name to `NUMERIC_FEATURES` (or appropriate list) and `FEATURE_DESCRIPTIONS`.

**2. Change model parameters:**
1.  Open `main.py`.
2.  Locate the `models` dictionary inside the `main()` function.
3.  Edit parameters (e.g., `n_estimators`, `max_depth`) directly there.

**3. Use your own data:**
1.  Make sure your Excel file follows the format expected (see `src/config.py` for column names).
2.  Run `main.py` with your file path:
    ```bash
    python main.py --data_path "path/to/your/file.xlsx"
    ```


