"""
compare_wizwhy.py
-----------------
Compares ML model predictions against WizWhy predictions on the shared test set.
Also computes SHAP feature importance for qualitative comparison with WizWhy rules.

Run:
    python scripts/compare_wizwhy.py
"""

import os
import sys
import joblib
import traceback

# Resolve paths relative to project root 
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data_loader import RealExcelDataLoader
from src.analysis.shap_explainability import compute_shap_summary, SHAP_AVAILABLE

THRESHOLD = 0.50

def recall_at_top_k(y_true, y_score, k_frac=0.20):
    df = pd.DataFrame({'true': y_true, 'score': y_score}).sort_values('score', ascending=False)
    top_n = max(1, int(len(df) * k_frac))
    captured = df.iloc[:top_n]['true'].sum()
    total = df['true'].sum()
    return captured / total if total > 0 else 0.0

def precision_at_top_k(y_true, y_score, k_frac=0.20):
    df = pd.DataFrame({'true': y_true, 'score': y_score}).sort_values('score', ascending=False)
    top_n = max(1, int(len(df) * k_frac))
    segment = df.iloc[:top_n]
    return segment['true'].sum() / top_n if top_n > 0 else 0.0

def compute_metrics(y_true, y_score, y_pred_binary=None, label=None, threshold=THRESHOLD):
    if y_pred_binary is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = y_pred_binary

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')

    return {
        'System': label,
        'AUC-ROC': round(auc, 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'Recall@Top20%': round(recall_at_top_k(y_true, y_score, 0.20), 4),
        'Precision@Top20%': round(precision_at_top_k(y_true, y_score, 0.20), 4),
    }

def main():
    print("=" * 70)
    print("WizWhy vs ML Model Comparison & SHAP Extraction")
    print("=" * 70)
    print(f"\nNOTE: Comparison threshold = {THRESHOLD}\n")

    # 1. Dataset Selection
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Error: data/ directory not found!")
        return
        
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not datasets:
        print("No dataset directories found in data/.")
        return
        
    print("\nSelect Dataset for Comparison")
    print("-" * 40)
    for i, d in enumerate(datasets, 1):
        print(f"  {i}. {d}")
        
    while True:
        try:
            choice = input(f"\nEnter the number of the dataset to compare [1-{len(datasets)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(datasets):
                selected_dataset = datasets[choice_idx]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

    dataset_config_map = {
        'first_file': {'employee_id_col': 'fictive2', 'time_col': 'fictive-ovedmiun'},
        'second_file': {'employee_id_col': 'fictive-oved', 'time_col': None},
        'factory_two': {'employee_id_col': 'fictive-oved', 'time_col': None},
    }
    dataset_config = dataset_config_map.get(selected_dataset, dataset_config_map['first_file'])
    EMPLOYEE_COL = dataset_config['employee_id_col']

    # 2. WizWhy Configuration Selection
    wizwhy_base = "files_wizwhy_relevant"
    wizwhy_folders = sorted([d for d in os.listdir(wizwhy_base) if os.path.isdir(os.path.join(wizwhy_base, d)) and d.startswith(selected_dataset)])
    
    if not wizwhy_folders:
        print(f"\nError: No WizWhy folders found in {wizwhy_base} starting with {selected_dataset}.")
        print("Please check your files_wizwhy_relevant folder.")
        return

    print(f"\nSelect WizWhy Run Configuration for {selected_dataset}")
    print("-" * 40)
    for i, d in enumerate(wizwhy_folders, 1):
        print(f"  {i}. {d}")
    
    while True:
        try:
            choice = input(f"\nEnter choice [1-{len(wizwhy_folders)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(wizwhy_folders):
                selected_wizwhy_folder = wizwhy_folders[choice_idx]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Invalid input.")

    # 3. Paths Configuration
    folder_path = os.path.join(wizwhy_base, selected_wizwhy_folder)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') or f.endswith('.csv')]
    
    if not txt_files:
        print(f"\nError: No WizWhy prediction .txt or .csv found inside {folder_path}!")
        return
        
    WIZWHY_PRED_PATH = os.path.join(folder_path, txt_files[0])
    print(f"\n[+] Auto-selected WizWhy results: {txt_files[0]}")

    ML_PRED_PATH  = os.path.join("output", f"predictions_{selected_dataset}_preset.xlsx")
    PIPELINE_PATH = os.path.join("artifacts", f"model_pipeline_{selected_dataset}_preset.pkl")
    RAW_DATA_PATH = os.path.join("data", selected_dataset, f"{selected_dataset}.xlsx")
    TEST_DATA_PATH = os.path.join("data", selected_dataset, "test_data.xlsx")
    
    # Mode naming for output
    mode_name = selected_wizwhy_folder.replace(selected_dataset, "").strip(" -_")
    safe_mode_name = mode_name.replace(" ", "_").lower() if mode_name else "default"
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_PATH = os.path.join(output_dir, f"shap_and_wizwhy_comparison_{selected_dataset}_{safe_mode_name}.xlsx")

    # Variables to track progress
    merged = pd.DataFrame()
    metrics_df = pd.DataFrame()

    # 4. Load WizWhy Predictions
    print("\n[1] Loading WizWhy Predictions...")
    try:
        try:
            ww = pd.read_csv(WIZWHY_PRED_PATH)
        except:
            ww = pd.read_csv(WIZWHY_PRED_PATH, sep='\t')
            
        ww.columns = ww.columns.str.strip().str.replace('"', '')
        ww_emp_col = 'fictive2' if 'fictive2' in ww.columns else 'fictive-oved'
        
        if ww_emp_col not in ww.columns:
            print(f"  Error: Could not find employee ID column ({ww_emp_col}) in WizWhy file.")
            print(f"  Available columns: {list(ww.columns)}")
            return

        ww[EMPLOYEE_COL] = pd.to_numeric(ww[ww_emp_col], errors='coerce').fillna(-1).astype(int).astype(str)
        2
        
        prob_col = 'Concl_Prob' if 'Concl_Prob' in ww.columns else 'Probability'
        if prob_col in ww.columns:
            # Shift " No Prediction" garbage text safely to 0.0 probability to preserve the metric scoring
            ww['ww_prob'] = pd.to_numeric(ww[prob_col], errors='coerce').fillna(0.0)
        else:
            ww['ww_prob'] = np.nan
            
        actual_col = 'leave_ind' if 'leave_ind' in ww.columns else 'Actual'
        if actual_col in ww.columns:
            ww['ww_actual'] = pd.to_numeric(ww[actual_col], errors='coerce').fillna(0).astype(int)
        else:
            ww['ww_actual'] = np.nan

        pred_col = 'Prediction' if 'Prediction' in ww.columns else 'Pred'
        if pred_col in ww.columns:
            ww['ww_pred'] = ww[pred_col].astype(str).str.strip().str.lower().apply(
                lambda x: 1 if 'more than' in x and 'no ' not in x else 0
            )
        else:
            ww['ww_pred'] = (ww['ww_prob'] >= 0.5).astype(int)

        ww = ww[[EMPLOYEE_COL, 'ww_actual', 'ww_prob', 'ww_pred']].drop_duplicates(EMPLOYEE_COL)
        print(f"  ✓ Loaded: {len(ww)} employees from WizWhy text file.")
        wizwhy_loaded = True
    except Exception as e:
        print(f"  Error parsing WizWhy file: {e}")
        traceback.print_exc()
        wizwhy_loaded = False

    # 5. Load ML Predictions
    print("\n[2] Loading ML Predictions...")
    if os.path.exists(ML_PRED_PATH):
        ml = pd.read_excel(ML_PRED_PATH)
        ml[EMPLOYEE_COL] = ml['Employee ID'].astype(float).astype(int).astype(str)
        ml['ml_prob'] = ml['Turnover Probability'].astype(float)
        ml['ml_actual'] = ml['Left Company (Actual)'].astype(float).astype(int)
        ml = ml[[EMPLOYEE_COL, 'ml_actual', 'ml_prob']].drop_duplicates(EMPLOYEE_COL)

        if os.path.exists(TEST_DATA_PATH):
            test_df = pd.read_excel(TEST_DATA_PATH)
            test_ids = set(test_df[EMPLOYEE_COL].astype(float).astype(int).astype(str))
            ml = ml[ml[EMPLOYEE_COL].isin(test_ids)]
            print(f"  ✓ Loaded: {len(ml)} test set employees.")
        else:
            print(f"  ✓ Loaded: {len(ml)} employees (Test masking disabled).")
        
        ml_loaded = True
    else:
        print(f"  File not found: {ML_PRED_PATH}")
        print("  Make sure you ran `python main.py` with the preset split to generate ML predictions first.")
        ml_loaded = False

    # 6. Compute Head-to-Head Metrics if both are loaded
    if wizwhy_loaded and ml_loaded:
        print("\n[3] Calculating Head-to-Head Metrics...")
        merged = pd.merge(ww, ml, on=EMPLOYEE_COL, how='inner')
        print(f"  Matched employees (inner join): {len(merged)}")

        if not merged.empty:
            valid_actuals = merged.dropna(subset=['ww_actual', 'ml_actual'])
            label_mismatch = (valid_actuals['ww_actual'] != valid_actuals['ml_actual']).sum()
            if label_mismatch > 0:
                print(f"  Warning: {label_mismatch} employees have mismatched actual labels.")

            y_true = merged['ml_actual'].values

            ww_metrics = compute_metrics(
                y_true, merged['ww_prob'].values, y_pred_binary=merged['ww_pred'].values, label='WizWhy'
            )
            ml_metrics = compute_metrics(
                y_true, merged['ml_prob'].values, y_pred_binary=None, label='ML Model', threshold=THRESHOLD
            )

            metrics_df = pd.DataFrame([ww_metrics, ml_metrics]).set_index('System')
            print("\nPerformance Comparison:")
            print(metrics_df.to_string())
    else:
        print("\n[3] Skipping Metrics calculation (missing prediction files).")

    # 7. SHAP feature extraction
    print("\n[4] Computing ML SHAP Values...")
    shap_df = pd.DataFrame()
    if not SHAP_AVAILABLE:
        print("  shap not installed — skipping. Run: pip install shap")
    elif not os.path.exists(PIPELINE_PATH):
        print(f"  Pipeline artifact not found at {PIPELINE_PATH} — skipping SHAP.")
    elif not os.path.exists(RAW_DATA_PATH):
        print(f"  Raw data not found at {RAW_DATA_PATH} — skipping SHAP.")
    else:
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            best_model = pipeline['model']
            scaler = pipeline['scaler']
            feature_names = pipeline['feature_names']

            loader = RealExcelDataLoader(RAW_DATA_PATH, **dataset_config)
            df_raw = loader.load()
            loader.scaler = scaler
            loader.feature_names = feature_names
            df_proc = loader.preprocess(df_raw, is_inference=True)
            X_all = df_proc[feature_names]

            kept_ids = [str(int(float(eid))) for eid in loader.get_kept_indices()]
            if os.path.exists(TEST_DATA_PATH):
                test_df = pd.read_excel(TEST_DATA_PATH)
                test_ids = set(test_df[EMPLOYEE_COL].astype(float).astype(int).astype(str))
                test_mask = [eid in test_ids for eid in kept_ids]
                X_shap = X_all[test_mask]
            else:
                X_shap = X_all

            print(f"  Computing SHAP on {len(X_shap)} test employees...")
            shap_df = compute_shap_summary(best_model, X_shap, feature_names, top_n=30)
            print("  ✓ SHAP extraction complete.")
        except Exception as e:
            print(f"  SHAP computation failed: {e}")

    # 8. Save to Excel
    print(f"\n[5] Saving results to {OUTPUT_PATH}")
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        if not metrics_df.empty:
            metrics_df.reset_index().to_excel(writer, sheet_name='Metrics Comparison', index=False)
            
            per_emp = merged[[EMPLOYEE_COL, 'ml_actual', 'ww_prob', 'ww_pred', 'ml_prob']].copy()
            per_emp.columns = ['Employee ID', 'Actual (leave_ind)', 'WizWhy Prob', 'WizWhy Pred', 'ML Prob']
            per_emp['ML Pred (≥0.50)'] = (per_emp['ML Prob'] >= THRESHOLD).astype(int)
            per_emp['Agreement'] = (per_emp['WizWhy Pred'] == per_emp['ML Pred (≥0.50)']).map({True: 'Yes', False: 'No'})
            per_emp = per_emp.sort_values('ML Prob', ascending=False)
            per_emp.to_excel(writer, sheet_name='Per-Employee Predictions', index=False)
            
        if not shap_df.empty:
            shap_df.to_excel(writer, sheet_name='SHAP Feature Importance', index=False)
            
        if metrics_df.empty and shap_df.empty:
            pd.DataFrame({'Warning': ['No data to save']}).to_excel(writer, sheet_name='Empty', index=False)

    print(f"\nDone. Saved to {OUTPUT_PATH}.")

if __name__ == "__main__":
    main()
