"""
compare_wizwhy.py
-----------------
Compares ML model predictions against WizWhy predictions on the shared test set.
Also computes SHAP feature importance for qualitative comparison with WizWhy rules.

Prerequisites:
  1. python create_split.py                   -- creates split/ folder
  2. WizWhy re-run on split/test_data.xlsx    -- new prediction txt placed in WizWhy/
  3. python main.py --split_file split/       -- retrains ML and saves artifacts

Run:
    python compare_wizwhy.py
"""

import os
import sys
import joblib

# Resolve paths relative to project root 
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

# Ensure console can handle Unicode (e.g. SHAP's progress arrows on Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data_loader import RealExcelDataLoader
from src.config import FEATURE_DESCRIPTIONS
from src.analysis.shap_explainability import compute_shap_summary, SHAP_AVAILABLE

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
WIZWHY_PRED_PATH   = os.path.join("WizWhy", "wizwhy_test_results.txt")
ML_PRED_PATH       = os.path.join("output", "predictions_first_file.xlsx")
PIPELINE_PATH      = os.path.join("artifacts", "model_pipeline_first_file_random.pkl")
RAW_DATA_PATH      = os.path.join("data", "first_file", "first_file.xlsx")
TEST_DATA_PATH     = os.path.join("data", "first_file", "test_data.xlsx")
OUTPUT_PATH        = os.path.join("WizWhy", "comparison_results.xlsx")

THRESHOLD          = 0.50
EMPLOYEE_COL       = "fictive2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """
    Compute metrics for a system.

    Args:
        y_true: Ground truth labels
        y_score: Probability scores (for AUC and ranking metrics)
        y_pred_binary: Pre-computed binary predictions (e.g., WizWhy's rule output).
                      If None, compute from y_score using threshold.
        label: System name
        threshold: Decision threshold (only used if y_pred_binary is None)
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("WizWhy vs ML Model Comparison")
    print("=" * 70)
    print(f"\nNOTE: Comparison threshold = {THRESHOLD}\n")

    # ------------------------------------------------------------------
    # 1. Load WizWhy predictions
    # ------------------------------------------------------------------
    if not os.path.exists(WIZWHY_PRED_PATH):
        print(f"Error: WizWhy predictions not found at {WIZWHY_PRED_PATH}")
        print("  Run WizWhy on split/test_data.xlsx and save the prediction txt there.")
        return

    ww = pd.read_csv(WIZWHY_PRED_PATH)
    ww.columns = ww.columns.str.strip().str.replace('"', '')
    ww_emp_col = 'fictive2' if 'fictive2' in ww.columns else 'fictive-oved'
    ww[EMPLOYEE_COL] = ww[ww_emp_col].astype(float).astype(int).astype(str)
    ww['ww_prob'] = ww['Concl_Prob'].astype(float)
    ww['ww_actual'] = ww['leave_ind'].astype(float).astype(int)

    # Parse WizWhy's Prediction column directly 
    # "more than 0.50" or similar → 1 (churn), "No more than 0.50" → 0 (no churn)
    ww['ww_pred'] = ww['Prediction'].str.strip().str.lower().apply(
        lambda x: 1 if 'more than' in x and 'no ' not in x else 0
    )

    ww = ww[[EMPLOYEE_COL, 'ww_actual', 'ww_prob', 'ww_pred']].drop_duplicates(EMPLOYEE_COL)
    print(f"WizWhy predictions loaded: {len(ww)} employees")

    # ------------------------------------------------------------------
    # 2. Load ML predictions (test set employees only)
    # ------------------------------------------------------------------
    if not os.path.exists(ML_PRED_PATH):
        print(f"Error: ML predictions not found at {ML_PRED_PATH}")
        print("  Run: python main.py --split_file split/")
        return

    ml = pd.read_excel(ML_PRED_PATH)
    ml[EMPLOYEE_COL] = ml['Employee ID'].astype(float).astype(int).astype(str)
    ml['ml_prob'] = ml['Turnover Probability'].astype(float)
    ml['ml_actual'] = ml['Left Company (Actual)'].astype(float).astype(int)
    ml = ml[[EMPLOYEE_COL, 'ml_actual', 'ml_prob']].drop_duplicates(EMPLOYEE_COL)

    # Filter to test employees only using test_data.xlsx
    if os.path.exists(TEST_DATA_PATH):
        test_df = pd.read_excel(TEST_DATA_PATH)
        actual_emp_col = 'fictive2' if 'fictive2' in test_df.columns else 'fictive-oved'
        test_ids = set(test_df[actual_emp_col].astype(float).astype(int).astype(str))
        ml = ml[ml[EMPLOYEE_COL].isin(test_ids)]
        print(f"ML predictions (test set only): {len(ml)} employees")
    else:
        print(f"ML predictions loaded: {len(ml)} employees (test_data.xlsx not found)")

    # ------------------------------------------------------------------
    # 3. Inner join on employee ID
    # ------------------------------------------------------------------
    merged = pd.merge(ww, ml, on=EMPLOYEE_COL, how='inner')
    print(f"\nMatched employees (inner join): {len(merged)}")

    if merged.empty:
        print("Error: No employees matched between WizWhy and ML predictions.")
        return

    # Sanity check: actual labels should agree
    label_mismatch = (merged['ww_actual'] != merged['ml_actual']).sum()
    if label_mismatch > 0:
        print(f"  Warning: {label_mismatch} employees have mismatched actual labels between sources.")

    y_true = merged['ml_actual'].values  # use ML source as ground truth

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Performance Comparison")
    print("-" * 70)
    print("Note: WizWhy metrics use its Prediction column (rule-based output).")
    print("      ML metrics use threshold=0.50 on probability scores.\n")

    ww_metrics = compute_metrics(
        y_true,
        merged['ww_prob'].values,
        y_pred_binary=merged['ww_pred'].values,
        label='WizWhy'
    )
    ml_metrics = compute_metrics(
        y_true,
        merged['ml_prob'].values,
        y_pred_binary=None,  # compute from threshold
        label='ML Model',
        threshold=THRESHOLD
    )

    metrics_df = pd.DataFrame([ww_metrics, ml_metrics]).set_index('System')
    print(metrics_df.to_string())

    # ------------------------------------------------------------------
    # 5. SHAP explainability
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ML Model Explainability (SHAP)")
    print("-" * 70)

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

            loader = RealExcelDataLoader(RAW_DATA_PATH)
            df_raw = loader.load()
            loader.scaler = scaler
            loader.feature_names = feature_names
            df_proc = loader.preprocess(df_raw, is_inference=True)
            X_all = df_proc[feature_names]

            # Filter to test employees for SHAP
            kept_ids = [str(int(float(eid))) for eid in loader.get_kept_indices()]
            if os.path.exists(TEST_DATA_PATH):
                test_df = pd.read_excel(TEST_DATA_PATH)
                actual_emp_col = 'fictive2' if 'fictive2' in test_df.columns else 'fictive-oved'
                test_ids = set(test_df[actual_emp_col].astype(float).astype(int).astype(str))
                test_mask = [eid in test_ids for eid in kept_ids]
                X_shap = X_all[test_mask]
            else:
                X_shap = X_all

            print(f"  Computing SHAP on {len(X_shap)} test employees...")
            shap_df = compute_shap_summary(best_model, X_shap, feature_names, top_n=20)

            print("\n  Top 20 Risk Drivers (compare with WizWhy if-then rules):")
            print(f"  {'Rank':<4} {'Feature':<40} {'Mean |SHAP|'}")
            print("  " + "-" * 60)
            for _, row in shap_df.iterrows():
                name = row['Readable_Name']
                if len(name) > 38:
                    name = name[:35] + "..."
                print(f"  {int(row['Rank']):<4} {name:<40} {row['Mean_Abs_SHAP']:.4f}")
        except Exception as e:
            print(f"  SHAP computation failed: {e}")

    # ------------------------------------------------------------------
    # 6. Save to Excel
    # ------------------------------------------------------------------
    os.makedirs("WizWhy", exist_ok=True)

    per_emp = merged[[EMPLOYEE_COL, 'ml_actual', 'ww_prob', 'ww_pred', 'ml_prob']].copy()
    per_emp.columns = ['Employee ID', 'Actual (leave_ind)', 'WizWhy Prob', 'WizWhy Pred', 'ML Prob']
    per_emp['ML Pred (≥0.50)'] = (per_emp['ML Prob'] >= THRESHOLD).astype(int)
    per_emp['Agreement'] = (per_emp['WizWhy Pred'] == per_emp['ML Pred (≥0.50)']).map({True: 'Yes', False: 'No'})
    per_emp = per_emp.sort_values('ML Prob', ascending=False)

    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        metrics_df.reset_index().to_excel(writer, sheet_name='Metrics Comparison', index=False)
        if not shap_df.empty:
            shap_df.to_excel(writer, sheet_name='SHAP Feature Importance', index=False)
        per_emp.to_excel(writer, sheet_name='Per-Employee Predictions', index=False)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("  Sheets: 'Metrics Comparison', 'SHAP Feature Importance', 'Per-Employee Predictions'")
    print("\nDone.")


if __name__ == "__main__":
    main()
