"""
compare_models.py
-----------------
Compares ML model trained on random 60/20/20 split vs 70/30 stratified split.

Both models should have been trained first:
  1. python main.py                        # Random split → artifacts/model_pipeline_random.pkl
  2. python main.py --split_file split/   # Fixed split → artifacts/model_pipeline_split.pkl

This script loads both, evaluates on the shared test set (70/30 split),
and generates a detailed comparison report.

Run:
    python scripts/compare_models.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Resolve paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

from src.data_loader import RealExcelDataLoader
from src.config import TARGET_COL
from src.evaluator import Evaluator

# Paths
DATA_PATH = os.path.join("data", "first_file", "first_file.xlsx")
RANDOM_MODEL_PATH = os.path.join("artifacts", "model_pipeline_first_file_random.pkl")
SPLIT_MODEL_PATH = os.path.join("artifacts", "model_pipeline_first_file_fixed.pkl")
TEST_DATA_PATH = os.path.join("data", "first_file", "test_data.xlsx")
COMPARISON_OUTPUT = os.path.join("output", "model_comparison.xlsx")


def recall_at_top_k(y_true, y_score, k_frac=0.20):
    """Recall in top k% of population."""
    df = pd.DataFrame({'true': y_true, 'score': y_score}).sort_values('score', ascending=False)
    top_n = max(1, int(len(df) * k_frac))
    captured = df.iloc[:top_n]['true'].sum()
    total = df['true'].sum()
    return captured / total if total > 0 else 0.0


def precision_at_top_k(y_true, y_score, k_frac=0.20):
    """Precision in top k% of population."""
    df = pd.DataFrame({'true': y_true, 'score': y_score}).sort_values('score', ascending=False)
    top_n = max(1, int(len(df) * k_frac))
    segment = df.iloc[:top_n]
    return segment['true'].sum() / top_n if top_n > 0 else 0.0


def compute_metrics(y_true, y_score, label=None, threshold=0.50):
    """Compute evaluation metrics."""
    y_pred = (y_score >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')

    return {
        'Model': label,
        'AUC-ROC': round(auc, 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'Recall@Top20%': round(recall_at_top_k(y_true, y_score, 0.20), 4),
        'Precision@Top20%': round(precision_at_top_k(y_true, y_score, 0.20), 4),
    }


def main():
    print("=" * 80)
    print("MODEL COMPARISON: Random Split vs Stratified Split")
    print("=" * 80)

    # Check prerequisites
    if not os.path.exists(RANDOM_MODEL_PATH):
        print(f"Error: Random model not found at {RANDOM_MODEL_PATH}")
        print("  Run: python main.py")
        return

    if not os.path.exists(SPLIT_MODEL_PATH):
        print(f"Error: Split model not found at {SPLIT_MODEL_PATH}")
        print("  Run: python main.py --split_file split/")
        return

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        print("  Please ensure test_data.xlsx exists in data folder.")
        return

    # Load models
    print("\n[1] Loading Models...")
    print("-" * 40)

    random_pipeline = joblib.load(RANDOM_MODEL_PATH)
    random_model = random_pipeline['model']
    random_scaler = random_pipeline['scaler']
    random_features = random_pipeline['feature_names']
    print(f"  ✓ Random split model loaded")

    split_pipeline = joblib.load(SPLIT_MODEL_PATH)
    split_model = split_pipeline['model']
    split_scaler = split_pipeline['scaler']
    split_features = split_pipeline['feature_names']
    print(f"  ✓ Stratified split model loaded")

    # Load and preprocess data
    print("\n[2] Loading and Preprocessing Data...")
    print("-" * 40)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    loader = RealExcelDataLoader(DATA_PATH)
    df_raw = loader.load()
    df = loader.preprocess(df_raw)

    # Load test IDs from test_data.xlsx
    test_df = pd.read_excel(TEST_DATA_PATH)
    EMPLOYEE_COL = 'fictive2' if 'fictive2' in test_df.columns else 'fictive-oved'
    test_ids = set(test_df[EMPLOYEE_COL].astype(float).astype(int).astype(str))

    # Filter to test set
    kept_ids = [str(int(float(eid))) for eid in loader.get_kept_indices()]
    test_mask = [eid in test_ids for eid in kept_ids]

    X_test = df[test_mask].drop(columns=[TARGET_COL], errors='ignore')
    y_test = df[test_mask][TARGET_COL]

    print(f"  Loaded test set: {len(X_test)} employees")
    print(f"  Churn rate: {y_test.mean()*100:.1f}%")

    # Get predictions from both models
    print("\n[3] Generating Predictions...")
    print("-" * 40)

    try:
        y_prob_random = random_model.predict_proba(X_test)
        print(f"  ✓ Random model predictions generated")
    except Exception as e:
        print(f"  ✗ Error generating random model predictions: {e}")
        return

    try:
        y_prob_split = split_model.predict_proba(X_test)
        print(f"  ✓ Stratified split model predictions generated")
    except Exception as e:
        print(f"  ✗ Error generating split model predictions: {e}")
        return

    # Compute metrics
    print("\n[4] Computing Metrics...")
    print("-" * 40)

    metrics_random = compute_metrics(y_test.values, y_prob_random, label='Random (60/20/20)')
    metrics_split = compute_metrics(y_test.values, y_prob_split, label='Stratified (70/30)')

    # Display comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    metrics_df = pd.DataFrame([metrics_random, metrics_split]).set_index('Model')
    print("\n" + metrics_df.to_string())

    # Compute differences
    print("\n" + "-" * 80)
    print("DIFFERENCES (Stratified - Random):")
    print("-" * 80)

    for metric in ['AUC-ROC', 'F1', 'Precision', 'Recall', 'Recall@Top20%', 'Precision@Top20%']:
        diff = metrics_split[metric] - metrics_random[metric]
        pct_change = (diff / metrics_random[metric] * 100) if metrics_random[metric] != 0 else 0
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  {metric:20s}: {symbol} {diff:+.4f} ({pct_change:+.1f}%)")

    # Per-employee predictions comparison
    print("\n[5] Per-Employee Predictions...")
    print("-" * 40)

    per_emp = pd.DataFrame({
        'Employee ID': [str(int(float(eid))) for eid in kept_ids if eid in test_ids],
        'Actual': y_test.values,
        'Random Model Prob': y_prob_random,
        'Split Model Prob': y_prob_split,
    })

    per_emp['Random Pred (≥0.50)'] = (per_emp['Random Model Prob'] >= 0.50).astype(int)
    per_emp['Split Pred (≥0.50)'] = (per_emp['Split Model Prob'] >= 0.50).astype(int)
    per_emp['Pred Agreement'] = (per_emp['Random Pred (≥0.50)'] == per_emp['Split Pred (≥0.50)']).map({True: 'Yes', False: 'No'})
    per_emp['Prob Difference'] = abs(per_emp['Random Model Prob'] - per_emp['Split Model Prob'])

    # Sort by probability difference to show biggest divergences
    per_emp_sorted = per_emp.sort_values('Prob Difference', ascending=False)

    print(f"\n  Total employees in test set: {len(per_emp)}")
    print(f"  Prediction agreement: {(per_emp['Pred Agreement'] == 'Yes').sum() / len(per_emp) * 100:.1f}%")
    print(f"  Mean probability difference: {per_emp['Prob Difference'].mean():.4f}")
    print(f"  Max probability difference: {per_emp['Prob Difference'].max():.4f}")

    print("\n  Top 10 Divergences (largest prediction differences):")
    print("  " + "-" * 100)
    print(f"  {'Emp ID':<10} {'Actual':<8} {'Random Prob':<15} {'Split Prob':<15} {'Difference':<12} {'Agreement':<12}")
    print("  " + "-" * 100)

    for idx, row in per_emp_sorted.head(10).iterrows():
        print(f"  {row['Employee ID']:<10} {row['Actual']:<8} {row['Random Model Prob']:<15.4f} "
              f"{row['Split Model Prob']:<15.4f} {row['Prob Difference']:<12.4f} {row['Pred Agreement']:<12}")

    # Save detailed comparison to Excel
    print("\n[6] Saving Results...")
    print("-" * 40)

    os.makedirs("output", exist_ok=True)

    with pd.ExcelWriter(COMPARISON_OUTPUT, engine='openpyxl') as writer:
        # Sheet 1: Metrics comparison
        metrics_df.reset_index().to_excel(writer, sheet_name='Metrics Comparison', index=False)

        # Sheet 2: Per-employee predictions
        per_emp_sorted.to_excel(writer, sheet_name='Per-Employee Predictions', index=False)

        # Sheet 3: Summary statistics
        summary_stats = pd.DataFrame({
            'Metric': [
                'Prediction Agreement (%)',
                'Mean Probability Difference',
                'Max Probability Difference',
                'Employees with >0.10 prob diff',
                'Employees with >0.20 prob diff',
            ],
            'Value': [
                (per_emp['Pred Agreement'] == 'Yes').sum() / len(per_emp) * 100,
                per_emp['Prob Difference'].mean(),
                per_emp['Prob Difference'].max(),
                (per_emp['Prob Difference'] > 0.10).sum(),
                (per_emp['Prob Difference'] > 0.20).sum(),
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)

    print(f"\n  Saved comparison to: {COMPARISON_OUTPUT}")
    print("  Sheets: 'Metrics Comparison', 'Per-Employee Predictions', 'Summary Statistics'")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    auc_diff = metrics_split['AUC-ROC'] - metrics_random['AUC-ROC']
    if abs(auc_diff) < 0.01:
        print("\n  ✓ Both split strategies yield similar performance (AUC difference < 0.01)")
    elif auc_diff > 0:
        print(f"\n  ↑ Stratified split performs better (+{auc_diff:.4f} AUC)")
    else:
        print(f"\n  ↓ Random split performs better ({auc_diff:.4f} AUC)")

    if (per_emp['Pred Agreement'] == 'Yes').sum() / len(per_emp) > 0.95:
        print("  → Models make same predictions 95%+ of the time")
    else:
        print(f"  → Models diverge on {100 - (per_emp['Pred Agreement'] == 'Yes').sum() / len(per_emp) * 100:.1f}% of cases")

    print("\nDone.")


if __name__ == "__main__":
    main()
