"""
create_split.py
---------------
Generates a canonical 70/30 stratified train/test split by employee ID
and exports raw Excel files into the corresponding dataset directory.

Run once:
    python scripts/create_split.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve paths relative to project root 
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

TARGET_COL = "leave_ind"
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Borrow config map from main logic
dataset_config_map = {
    'first_file': {'employee_id_col': 'fictive2', 'time_col': 'fictive-ovedmiun'},
    'second_file': {'employee_id_col': 'fictive-oved', 'time_col': None},
    'factory_two': {'employee_id_col': 'fictive-oved', 'time_col': None},
}

def choose_dataset():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Error: data/ directory not found!")
        sys.exit(1)
        
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if not datasets:
        print("No dataset directories found in data/.")
        sys.exit(1)
        
    print("\n" + "="*40)
    print("Select Dataset to Split")
    print("="*40)
    for i, d in enumerate(datasets, 1):
        print(f"  {i}. {d}")
        
    while True:
        try:
            choice = input(f"\nEnter the number of the dataset to use [1-{len(datasets)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(datasets):
                return datasets[choice_idx]
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    dataset_tag = choose_dataset()
    dataset_folder = os.path.join("data", dataset_tag)
    
    # Locate raw file
    raw_files = [f for f in os.listdir(dataset_folder) if not f.startswith("train_") and not f.startswith("test_") and f.endswith(".xlsx")]
    if not raw_files:
        print(f"Error: No raw Excel file found in {dataset_folder}")
        return
        
    DATA_PATH = os.path.join(dataset_folder, raw_files[0])
    OUT_DIR = dataset_folder

    config = dataset_config_map.get(dataset_tag, dataset_config_map['first_file'])
    EMPLOYEE_COL = config['employee_id_col']
    TIME_COL = config['time_col']

    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"  Loaded {len(df)} total records, {df[EMPLOYEE_COL].nunique()} unique employees.")

    # One label per employee: use the last record's leave_ind
    if TIME_COL and TIME_COL in df.columns:
        emp_labels = (
            df.sort_values([EMPLOYEE_COL, TIME_COL])
            .groupby(EMPLOYEE_COL)[TARGET_COL]
            .last()
            .reset_index()
        )
    else:
        emp_labels = (
            df.groupby(EMPLOYEE_COL)[TARGET_COL]
            .last()
            .reset_index()
        )

    all_ids = emp_labels[EMPLOYEE_COL].values
    all_labels = emp_labels[TARGET_COL].values

    train_ids, test_ids, _, _ = train_test_split(
        all_ids, all_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_labels
    )

    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    # Verify stratification
    train_labels = emp_labels[emp_labels[EMPLOYEE_COL].isin(train_ids_set)][TARGET_COL]
    test_labels = emp_labels[emp_labels[EMPLOYEE_COL].isin(test_ids_set)][TARGET_COL]

    print(f"\nSplit complete (seed={RANDOM_STATE}, test_size={TEST_SIZE}):")
    print(f"  Train employees : {len(train_ids)} ({len(train_ids)/len(all_ids)*100:.1f}%)")
    print(f"    Churn rate    : {train_labels.mean()*100:.1f}%")
    print(f"  Test employees  : {len(test_ids)} ({len(test_ids)/len(all_ids)*100:.1f}%)")
    print(f"    Churn rate    : {test_labels.mean()*100:.1f}%")

    # Export raw Excel files (all time-period records per employee)
    train_df = df[df[EMPLOYEE_COL].isin(train_ids_set)].copy()
    test_df = df[df[EMPLOYEE_COL].isin(test_ids_set)].copy()

    train_xlsx = os.path.join(OUT_DIR, "train_data.xlsx")
    test_xlsx = os.path.join(OUT_DIR, "test_data.xlsx")

    train_df.to_excel(train_xlsx, index=False)
    test_df.to_excel(test_xlsx, index=False)

    print(f"\n  Saved {train_xlsx} ({len(train_df)} rows)")
    print(f"  Saved {test_xlsx} ({len(test_df)} rows)")

    print("\nDone.")
    print("Now run: python main.py --use_preset_split")


if __name__ == "__main__":
    main()
