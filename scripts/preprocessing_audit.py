import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_COL
from src.datasets import discover_dataset_specs, read_excel_with_header_detection
from src.static_preprocessing import build_static_model_frame


OUTPUT_DIR = Path("output/preprocessing_audit")


GLOBAL_FINDINGS = [
    {
        "Area": "Train/test leakage",
        "Finding": "RealExcelDataLoader currently imputes missing values, caps outliers, one-hot encodes categories, and fits MinMaxScaler before train/validation/test splitting.",
        "Impact": "Validation/test metrics can be optimistic because preprocessing learned from held-out data.",
        "Recommendation": "Move imputation, outlier handling, encoding, and scaling into an sklearn Pipeline fit only on training data.",
    },
    {
        "Area": "Time aggregation",
        "Finding": "For first_file, the legacy path creates last/mean/std/trend features and then predicts latest leave_ind.",
        "Impact": "Strong time-history features may be valid for a time-series goal, but they obscure which simple current-state features drive turnover.",
        "Recommendation": "Keep time-series and static/current-state pipelines separate and name outputs accordingly.",
    },
    {
        "Area": "Tracking count features",
        "Finding": "num_periods and data_maturity are derived from how many records exist for an employee.",
        "Impact": "They can dominate importance while not being an HR lever and may encode data collection history rather than employee risk.",
        "Recommendation": "Exclude them from static management/actionability reports.",
    },
    {
        "Area": "Outcome metadata",
        "Finding": "Third-source files include departure code/date/year style columns.",
        "Impact": "These are target leakage if included as predictors.",
        "Recommendation": "Drop exit-code/date/year fields before modeling.",
    },
    {
        "Area": "High-cardinality IDs/codes",
        "Finding": "Manager code, role code, city code, and budget section can be highly predictive.",
        "Impact": "They are useful for local diagnosis but may reduce portability to a general model.",
        "Recommendation": "Compare all-static, no-company-code, management-lever, and core-general feature sets.",
    },
]


def summarize_source(spec):
    raw, header_row = read_excel_with_header_detection(spec.path)
    static_frame, dropped_cols = build_static_model_frame(raw, spec)

    if TARGET_COL in raw.columns:
        raw_target = raw[TARGET_COL].dropna()
        raw_leave_rate = raw_target.mean() if len(raw_target) else None
    else:
        raw_leave_rate = None

    latest_target = static_frame[TARGET_COL].dropna()
    latest_leave_rate = latest_target.mean() if len(latest_target) else None

    rows = {
        "Dataset": spec.tag,
        "Source_File": str(spec.path),
        "Header_Row": header_row,
        "Employee_ID_Column": spec.employee_id_col,
        "Time_Column": spec.time_col or "",
        "Raw_Rows": len(raw),
        "Raw_Columns": len(raw.columns),
        "Unique_Employees_Raw": raw[spec.employee_id_col].nunique() if spec.employee_id_col in raw.columns else None,
        "Latest_Static_Rows": len(static_frame),
        "Latest_Static_Columns": len(static_frame.columns),
        "Raw_Leave_Rate": raw_leave_rate,
        "Latest_Leave_Rate": latest_leave_rate,
        "Dropped_Leakage_Time_Columns": ", ".join(str(c) for c in dropped_cols),
    }

    missing_rows = []
    for col in static_frame.columns:
        missing_pct = static_frame[col].isna().mean()
        if missing_pct > 0:
            missing_rows.append(
                {
                    "Dataset": spec.tag,
                    "Column": col,
                    "Missing_Pct": missing_pct,
                    "Missing_Count": int(static_frame[col].isna().sum()),
                }
            )

    cardinality_rows = []
    for col in static_frame.select_dtypes(include=["object"]).columns:
        cardinality_rows.append(
            {
                "Dataset": spec.tag,
                "Column": col,
                "Unique_Values": int(static_frame[col].nunique(dropna=True)),
            }
        )

    return rows, missing_rows, cardinality_rows


def write_markdown(source_df, finding_df, skipped_df, path):
    lines = [
        "# Preprocessing Audit",
        "",
        "This audit separates code-quality/preprocessing risks from model results.",
        "",
        "## Main Findings",
        "",
    ]

    for _, row in finding_df.iterrows():
        lines.extend(
            [
                f"### {row['Area']}",
                "",
                f"- Finding: {row['Finding']}",
                f"- Impact: {row['Impact']}",
                f"- Recommendation: {row['Recommendation']}",
                "",
            ]
        )

    lines.extend(["## Source Inventory", ""])
    lines.append("| Dataset | Raw rows | Latest employees | Raw leave rate | Latest leave rate | Dropped leakage/time columns |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for _, row in source_df.iterrows():
        raw_rate = "" if pd.isna(row["Raw_Leave_Rate"]) else f"{row['Raw_Leave_Rate']:.1%}"
        latest_rate = "" if pd.isna(row["Latest_Leave_Rate"]) else f"{row['Latest_Leave_Rate']:.1%}"
        lines.append(
            f"| {row['Dataset']} | {int(row['Raw_Rows'])} | {int(row['Latest_Static_Rows'])} | "
            f"{raw_rate} | {latest_rate} | {row['Dropped_Leakage_Time_Columns']} |"
        )

    lines.extend(["", "## Skipped Sources", ""])
    if skipped_df.empty:
        lines.append("No discovered source files were skipped.")
    else:
        for _, row in skipped_df.iterrows():
            lines.append(f"- `{row['Source_File']}`: {row['Reason']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset schemas and preprocessing risks.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs, skipped = discover_dataset_specs()
    source_rows = []
    missing_rows = []
    cardinality_rows = []

    for spec in specs:
        print(f"Auditing {spec.tag}...")
        try:
            source, missing, cardinality = summarize_source(spec)
            source_rows.append(source)
            missing_rows.extend(missing)
            cardinality_rows.extend(cardinality)
        except Exception as exc:
            skipped.append({"Dataset": spec.tag, "Source_File": str(spec.path), "Reason": str(exc)})
            print(f"  skipped: {exc}")

    source_df = pd.DataFrame(source_rows).sort_values("Dataset")
    finding_df = pd.DataFrame(GLOBAL_FINDINGS)
    missing_df = pd.DataFrame(missing_rows).sort_values(["Dataset", "Missing_Pct"], ascending=[True, False])
    cardinality_df = pd.DataFrame(cardinality_rows).sort_values(["Dataset", "Unique_Values"], ascending=[True, False])
    skipped_df = pd.DataFrame(skipped)

    excel_path = output_dir / "preprocessing_audit.xlsx"
    md_path = output_dir / "README.md"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        finding_df.to_excel(writer, sheet_name="Main Findings", index=False)
        source_df.to_excel(writer, sheet_name="Source Inventory", index=False)
        missing_df.to_excel(writer, sheet_name="Missing Values", index=False)
        cardinality_df.to_excel(writer, sheet_name="Categorical Cardinality", index=False)
        skipped_df.to_excel(writer, sheet_name="Skipped Sources", index=False)

    write_markdown(source_df, finding_df, skipped_df, md_path)
    print(f"Audit workbook: {excel_path}")
    print(f"Audit summary:  {md_path}")


if __name__ == "__main__":
    main()
