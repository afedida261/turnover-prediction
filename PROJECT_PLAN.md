# Turnover Prediction Project Plan (Hilan)

Based on `094395-6 Requirements Document V1.2.1.pdf` and `שדות.xlsx`.

## 1. Project Architecture & Modularity

The project will follow a modular design to allow easy swapping of models and feature sets.

```
turnover-prediction/
├── data/
│   ├── raw/ (Generated CSVs with Hebrew headers)
│   └── processed/ (Cleaned data with English headers)
├── src/
│   ├── __init__.py
│   ├── config.py           # Feature mappings, constants, thresholds
│   ├── data_loader.py      # Loads CSV, renames columns, handles types
│   ├── preprocessor.py     # Cleaning, encoding, scaling
│   ├── models/
│   │   ├── base_model.py   # Abstract Base Class
│   │   ├── classifiers.py  # RF, XGB, Logistic
│   │   ├── nn_model.py     # PyTorch Neural Network
│   │   └── survival.py     # CoxPH Model
│   ├── evaluator.py        # Metrics (Recall@Top20%, Lift, etc.)
│   └── pipeline.py         # Orchestrator
├── main.py                 # Entry point
├── generate_data.py        # Synthetic data generator matching 'שדות.xlsx'
└── requirements.txt
```

## 2. Data Generation (`generate_data.py`)

We will generate a CSV file `data/raw/hilan_synthetic_data.csv` with the exact columns found in `שדות.xlsx`.

**Column Mapping (Hebrew -> Internal English):**
*   `אינדיקציה לעזיבה - המשתנה התלוי` -> `Target_Churn`
*   `ותק בחודשים` -> `Tenure_Months`
*   `גיל` -> `Age`
*   `מגדר` -> `Gender`
*   `מצב משפחתי` -> `Marital_Status`
*   `זמן נסיעה לעבודה` -> `Commute_Time_Est`
*   `יישוב מגורים` -> `Residence_City`
*   `סעיף תקציבי/שיוך ארגוני` -> `Organizational_Unit`
*   `מעמד` -> `Job_Rank`
*   `דירוג` -> `Job_Rating`
*   `דגל תפקיד` -> `Flag_Role_Stagnation`
*   `דגל מנהל` -> `Flag_Manager_Change`
*   `דגל ניצול מחלה` -> `Flag_Sick_Abuse` (or similar)
*   `דגל שעות עבודה` -> `Flag_Annual_Hours_Change`
*   `דגל שכר` -> `Flag_Salary_Freeze`
*   `דגל מצב אישי` -> `Flag_Personal_Change`
*   `avg_Payment` -> `Salary_Avg_12m`
*   `stdevp_Payment` -> `Salary_StDev`
*   `Median_Payment` -> `Salary_Median`
*   `שינוי השכר בשקלים` -> `Salary_Change_Value`
*   `avg_illness` -> `Sick_Days_Avg`
*   `stdevp_illness` -> `Sick_Days_StDev`
*   `Median_illness` -> `Sick_Days_Median`
*   `avg_omes` -> `Work_Load_Avg`
*   `stdevp_omes` -> `Work_Load_StDev`
*   `Median_omes` -> `Work_Load_Median`
*   `count_managers` -> `Manager_Count_Total`

## 3. Feature Engineering & Preprocessing

*   **Snapshot Strategy**: The data generation will simulate the T-12 snapshot logic.
*   **Imbalance Handling**: Implement SMOTE or Class Weighting (as per section 3.2.1.2).
*   **Encoding**: One-Hot Encoding for `Gender`, `Marital_Status`, `Residence_City`.

## 4. Model Implementation

*   **Voting Classifier**: Combine XGBoost, Random Forest, and Neural Network.
*   **Survival Analysis**: Cox Proportional Hazards to predict `Time_To_Event`.
    *   *Note*: The raw data needs a `Months_Until_Event` column for training the survival model, even if it's not in the standard input list for the classifier. We will generate this as a hidden target column.

## 5. Evaluation Metrics (Section 3.2.2.2)

*   **Recall@Top20%**: Must be >= 70%.
*   **AUC-ROC**: Secondary metric.
*   **Lift**: Optional but useful.

## 6. Step-by-Step Execution Plan

1.  **Setup**: Create directory structure and `config.py`.
2.  **Data**: Implement `generate_data.py` to create the Hebrew CSV.
3.  **Loader**: Implement `data_loader.py` to read Hebrew CSV and map to English.
4.  **Models**: Refactor existing models into the new structure.
5.  **Pipeline**: Connect Loader -> Preprocessor -> Model -> Evaluator.
6.  **Run**: Execute `main.py` to train and generate the report.
