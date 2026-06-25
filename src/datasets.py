from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import TARGET_COL


EMPLOYEE_ID_CANDIDATES = ["fictive2", "fictive-oved", "fictive_employee"]


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    path: Path
    employee_id_col: str
    time_col: Optional[str] = None
    header_row: int = 0
    source_kind: str = "excel"


KNOWN_DATASET_DEFAULTS = {
    "file1": {"employee_id_col": "fictive_employee", "time_col": "calc_month"},
    "file2": {"employee_id_col": "fictive_employee", "time_col": "calc_month"},
    "first_file": {"employee_id_col": "fictive2", "time_col": "fictive-ovedmiun"},
    "second_file": {"employee_id_col": "fictive-oved", "time_col": None},
    "factory_two": {"employee_id_col": "fictive-oved", "time_col": None},
    "file3": {"employee_id_col": "fictive_employee", "time_col": "calc_month"},
    "ml3": {"employee_id_col": "fictive_employee", "time_col": "calc_month"},
    "file3-toStudents": {"employee_id_col": "fictive_employee", "time_col": "calc_month"},
    "ML-3-original -no PII-1": {"employee_id_col": "fictive_employee", "time_col": "calc_month"},
}


def read_excel_with_header_detection(path: str | Path, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, int]:
    path = Path(path)
    df = pd.read_excel(path)
    if target_col in df.columns:
        return df, 0

    probe = pd.read_excel(path, header=None, nrows=5)
    for row_idx in range(len(probe)):
        values = set(str(v) for v in probe.iloc[row_idx].dropna().tolist())
        if target_col in values:
            return pd.read_excel(path, header=row_idx), row_idx

    return df, 0


def infer_employee_id_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in EMPLOYEE_ID_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def infer_time_col(df: pd.DataFrame, dataset_tag: str) -> Optional[str]:
    configured = KNOWN_DATASET_DEFAULTS.get(dataset_tag, {}).get("time_col")
    if configured and configured in df.columns:
        return configured
    for candidate in ["fictive-ovedmiun", "calc_month", "year_date"]:
        if candidate in df.columns:
            return candidate
    return None


def spec_for_path(path: str | Path, dataset_tag: Optional[str] = None) -> DatasetSpec:
    path = Path(path)
    tag = dataset_tag or path.stem
    df, header_row = read_excel_with_header_detection(path)
    defaults = KNOWN_DATASET_DEFAULTS.get(tag, {})
    employee_id_col = defaults.get("employee_id_col")
    if not employee_id_col or employee_id_col not in df.columns:
        employee_id_col = infer_employee_id_col(df)
    if employee_id_col is None:
        raise ValueError(f"Could not infer employee ID column for {path}")

    time_col = infer_time_col(df, tag)
    return DatasetSpec(
        tag=tag,
        path=path,
        employee_id_col=employee_id_col,
        time_col=time_col,
        header_row=header_row,
        source_kind="data_excel" if path.parent == Path("data") else "data_folder",
    )


def discover_dataset_specs(include_root_excels: bool = True, include_data_excels: bool = True) -> tuple[list[DatasetSpec], list[dict]]:
    specs = []
    skipped = []

    data_dir = Path("data")
    if data_dir.exists():
        if include_data_excels:
            for path in sorted(data_dir.glob("file*.xlsx")):
                try:
                    specs.append(spec_for_path(path, path.stem))
                except Exception as exc:
                    skipped.append({"Dataset": path.stem, "Source_File": str(path), "Reason": str(exc)})

        for folder in sorted(data_dir.iterdir()):
            if not folder.is_dir() or folder.name.lower() in {"raw", "old"}:
                continue
            raw_files = [
                p for p in sorted(folder.glob("*.xlsx"))
                if not p.name.startswith("train_") and not p.name.startswith("test_")
            ]
            if not raw_files:
                continue
            try:
                specs.append(spec_for_path(raw_files[0], folder.name))
            except Exception as exc:
                skipped.append({"Dataset": folder.name, "Source_File": str(raw_files[0]), "Reason": str(exc)})

    if include_root_excels:
        for path in sorted(Path(".").glob("*.xlsx")):
            try:
                specs.append(spec_for_path(path, path.stem))
            except Exception as exc:
                skipped.append({"Dataset": path.stem, "Source_File": str(path), "Reason": str(exc)})

    return specs, skipped


def normalize_employee_id(value) -> str:
    if pd.isna(value):
        return ""
    value_str = str(value).strip()
    try:
        value_float = float(value_str)
        if value_float.is_integer():
            return str(int(value_float))
    except ValueError:
        pass
    return value_str


def leakage_or_time_columns(
    df: pd.DataFrame,
    *,
    employee_id_col: str,
    target_col: str = TARGET_COL,
    time_col: Optional[str] = None,
) -> list[str]:
    protected_cols = {target_col, employee_id_col}
    drop_cols = []

    for col in df.columns:
        if col in protected_cols:
            continue

        col_text = str(col).lower()
        is_outcome_metadata = (
            "aziva" in col_text
            or "עזיב" in col_text
            or col_text == "target"
            or col_text.endswith("_target")
            or col_text in {"year_date", "calc_month"}
            or col_text.endswith("_date")
            or col_text.endswith("_year")
        )
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        is_time_col = bool(time_col and col == time_col)

        if is_outcome_metadata or is_datetime or is_time_col:
            drop_cols.append(col)

    return drop_cols


def drop_leakage_or_time_columns(
    df: pd.DataFrame,
    *,
    employee_id_col: str,
    target_col: str = TARGET_COL,
    time_col: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = leakage_or_time_columns(
        df,
        employee_id_col=employee_id_col,
        target_col=target_col,
        time_col=time_col,
    )
    return df.drop(columns=drop_cols, errors="ignore"), drop_cols
