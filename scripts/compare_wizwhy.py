"""
compare_wizwhy.py
-----------------
Computes SHAP feature importance for qualitative comparison with WizWhy rules.
Outputs results to files_wizwhy_relevant.

Run:
    python scripts/compare_wizwhy.py
"""

import os
import sys
import joblib
import pandas as pd

# Resolve paths relative to project root 
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.data_loader import RealExcelDataLoader
from src.analysis.shap_explainability import compute_shap_summary, SHAP_AVAILABLE

def main():
    print("=" * 70)
    print("Extracting SHAP values to compare with WizWhy Rules")
    print("=" * 70)

    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Error: data/ directory not found!")
        return
        
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not datasets:
        print("No dataset directories found in data/.")
        return
        
    print("\nSelect Dataset to generate SHAP comparisons for")
    print("-" * 40)
    for i, d in enumerate(datasets, 1):
        print(f"  {i}. {d}")
        
    while True:
        try:
            choice = input(f"\nEnter the number of the dataset [1-{len(datasets)}]: ")
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

    PIPELINE_PATH  = os.path.join("artifacts", f"model_pipeline_{selected_dataset}_preset.pkl")
    RAW_DATA_PATH  = os.path.join("data", selected_dataset, f"{selected_dataset}.xlsx")
    TEST_DATA_PATH = os.path.join("data", selected_dataset, "test_data.xlsx")
    
    # We output to the 'output' folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_PATH = os.path.join(output_dir, f"shap_comparison_{selected_dataset}.xlsx")

    if not SHAP_AVAILABLE:
        print("Error: shap not installed — please run: pip install shap")
        return
        
    if not os.path.exists(PIPELINE_PATH):
        print(f"Error: Pipeline artifact not found at {PIPELINE_PATH}.")
        print("Run 'python main.py' and choose the preset split to generate the model.")
        return
        
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data not found at {RAW_DATA_PATH}.")
        return

    print("\n[1] Loading Model Pipeline...")
    pipeline = joblib.load(PIPELINE_PATH)
    best_model = pipeline['model']
    scaler = pipeline['scaler']
    feature_names = pipeline['feature_names']

    print(f"\n[2] Loading & Preprocessing Test Data ({TEST_DATA_PATH})...")
    loader = RealExcelDataLoader(RAW_DATA_PATH, **dataset_config)
    df_raw = loader.load()
    loader.scaler = scaler
    loader.feature_names = feature_names
    df_proc = loader.preprocess(df_raw, is_inference=True)
    X_all = df_proc[feature_names]

    # Filter to test employees for SHAP
    kept_ids = [str(int(float(eid))) for eid in loader.get_kept_indices()]
    if os.path.exists(TEST_DATA_PATH):
        test_df = pd.read_excel(TEST_DATA_PATH)
        actual_emp_col = dataset_config['employee_id_col']
        test_ids = set(test_df[actual_emp_col].astype(float).astype(int).astype(str))
        test_mask = [eid in test_ids for eid in kept_ids]
        X_shap = X_all[test_mask]
    else:
        print("Warning: test_data.xlsx not found, using all data for SHAP.")
        X_shap = X_all

    print(f"\n[3] Computing SHAP on {len(X_shap)} test employees...")
    shap_df = compute_shap_summary(best_model, X_shap, feature_names, top_n=30)

    print("\nTop Risk Drivers (to compare with WizWhy Rules):")
    print(f"{'Rank':<4} {'Feature':<40} {'Mean |SHAP|'}")
    print("-" * 65)
    for _, row in shap_df.head(20).iterrows():
        name = row['Readable_Name']
        if len(name) > 38:
            name = name[:35] + "..."
        print(f"{int(row['Rank']):<4} {name:<40} {row['Mean_Abs_SHAP']:.4f}")

    print(f"\n[4] Saving results to {OUTPUT_PATH}")
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        shap_df.to_excel(writer, sheet_name='SHAP Feature Importance', index=False)

    print("\nDone. You can now use this Excel to manually cross-reference the WizWhy PDFs.")

if __name__ == "__main__":
    main()
