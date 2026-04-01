"""
create_split.py
---------------
Generates a canonical 70/30 stratified train/test split by employee ID
and exports raw Excel files for WizWhy and the ML pipeline.

Run once:
    python create_split.py

Outputs in split/:
    train_ids.txt       -- employee IDs (fictive2) for training
    test_ids.txt        -- employee IDs (fictive2) for testing
    train_data.xlsx     -- raw rows from first_file.xlsx for train employees
    test_data.xlsx      -- raw rows from first_file.xlsx for test employees
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve paths relative to project root 
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

DATA_PATH = os.path.join("data", "first_file.xlsx")
OUT_DIR = "split"
EMPLOYEE_COL = "fictive2"
TIME_COL = "fictive-ovedmiun"
TARGET_COL = "leave_ind"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print(f"Loading {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"  Loaded {len(df)} total records, {df[EMPLOYEE_COL].nunique()} unique employees.")

    # One label per employee: use the last record's leave_ind
    if TIME_COL in df.columns:
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

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save ID lists
    train_ids_path = os.path.join(OUT_DIR, "train_ids.txt")
    test_ids_path = os.path.join(OUT_DIR, "test_ids.txt")

    with open(train_ids_path, "w") as f:
        f.write("\n".join(str(i) for i in train_ids))
    with open(test_ids_path, "w") as f:
        f.write("\n".join(str(i) for i in test_ids))

    print(f"\n  Saved {train_ids_path}")
    print(f"  Saved {test_ids_path}")

    # Export raw Excel files (all time-period records per employee)
    train_df = df[df[EMPLOYEE_COL].isin(train_ids_set)].copy()
    test_df = df[df[EMPLOYEE_COL].isin(test_ids_set)].copy()

    train_xlsx = os.path.join(OUT_DIR, "train_data.xlsx")
    test_xlsx = os.path.join(OUT_DIR, "test_data.xlsx")

    train_df.to_excel(train_xlsx, index=False)
    test_df.to_excel(test_xlsx, index=False)

    print(f"  Saved {train_xlsx} ({len(train_df)} rows)")
    print(f"  Saved {test_xlsx} ({len(test_df)} rows)")

    print("\nDone. Feed split/train_data.xlsx and split/test_data.xlsx to WizWhy.")
    print("Then run: python main.py --split_file split/")


if __name__ == "__main__":
    main()
