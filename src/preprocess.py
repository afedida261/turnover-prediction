"""EDA-driven preprocessing for file1 + file2 training and file3 testing.

The module keeps employee-period rows, removes the invalid records confirmed by
the user, creates history features using only information available at or before
each row, and exposes a fold-safe sklearn preprocessing pipeline.

Important: ``prepare_turnover_data`` does not fit global imputers. During model
validation, call ``make_fold_safe_model_preprocessor`` inside the estimator
pipeline so payment imputation is refitted on every employee-grouped train fold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.imputations import PAYMENT_COLUMNS, SimilarEmployeePaymentImputer
from src.config import TARGET_COL
from src.datasets import read_excel_with_header_detection


DEFAULT_SOURCE_PATHS = {
    "file1": Path("data/file1.xlsx"),
    "file2": Path("data/file2.xlsx"),
    "file3": Path("data/file3.xlsx"),
}

TRAIN_SOURCES = {"file1", "file2"}
EXTERNAL_TEST_SOURCE = "file3"

COLUMN_ALIASES = {
    "TeurGroupHscm": "contract_type",
    "EMP_Matzav_Mishpachti": "marital_status",
    "שינוי השכר בשקלים": "salary_change",
}

CATEGORY_REPLACEMENTS = {
    "gender": {
        "גבר": "Male",
        "זכר": "Male",
        "אישה": "Female",
        "נקבה": "Female",
    },
    "contract_type": {
        "גלובלי": "Global",
        "חודשי גלובלי": "Global",
        "חודשי": "Monthly",
        "שעתי": "Hourly",
        "עובד חיצוני": "External",
    },
    "marital_status": {
        "נשוי/נשואה": "Married",
        "רווק/ה": "Single",
        "גרוש/ה": "Divorced",
        "אלמן/ה": "Widowed",
    },
}

OUTCOME_METADATA_COLUMNS = {
    "aziva_kod",
    "aziva_date",
    "aziva_year",
    "target",
}

# Retained in the frame long enough to support ordering/imputation, but never
# selected by the final ColumnTransformer as model inputs.
AUXILIARY_ONLY_COLUMNS = {
    "source",
    "source_employee_id",
    "fictive_employee",
    "calc_month",
    "year_date",
}

# EDA found reversed source-specific behavior for Tafkid and recommends keeping
# protected attributes descriptive unless a fairness rationale is documented.
DEFAULT_EXCLUDED_MODEL_COLUMNS = {
    "Derug",
    "Distance",
    "Tafkid",
    "gender",
    "marital_status",
}

HISTORY_NUMERIC_COLUMNS = (
    "avg_Payment",
    "Median_Payment",
    "stdevp_Payment",
    "salary_change",
    "avg_illness",
    "Median_illness",
    "stdevp_illness",
    "avg_omes",
    "Median_omes",
    "stdevp_omes",
    "count_managers",
)

HISTORY_CATEGORICAL_COLUMNS = (
    "contract_type",
    "manager_Code",
    "tafkidCode",
    "Maamad",
    "Maneger",
    "Mahala",
)


@dataclass(frozen=True)
class CleaningAudit:
    input_rows: int
    output_rows: int
    removed_rows: int
    removed_positive_rows: int
    retained_age_above_75_rows: int
    reason_counts: dict[str, int]
    reason_counts_by_source: dict[str, dict[str, int]]


@dataclass
class PreparedTurnoverData:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    train_groups: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_groups: pd.Series
    cleaning_audit: CleaningAudit
    dropped_model_columns: list[str]



def signed_log1p(values):
    """Signed log transform that is safe for negative salary changes."""
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.log1p(np.abs(array))


def canonicalize_columns_and_categories(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in frame.columns}).copy()
    for column, replacements in CATEGORY_REPLACEMENTS.items():
        if column in data.columns:
            data[column] = data[column].replace(replacements)
    return data


def validate_target_metadata(frame: pd.DataFrame) -> None:
    required = {TARGET_COL, "source", "fictive_employee"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    invalid_targets = set(pd.Series(frame[TARGET_COL]).dropna().unique()).difference({0, 1})
    if invalid_targets:
        raise ValueError(f"Unexpected target values: {sorted(invalid_targets)}")

    if "aziva_kod" in frame.columns:
        expected = frame["source"].map({"file1": 41, "file2": 41, "file3": 42})
        positives = frame[TARGET_COL].eq(1)
        mismatched = positives & ~frame["aziva_kod"].eq(expected)
        if mismatched.any():
            raise ValueError(f"{int(mismatched.sum())} positive rows have an unexpected Aziva code.")


def clean_problematic_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
    """Remove user-confirmed invalid rows while retaining employees over 75."""
    data = frame.copy()
    conditions = {
        "negative_average_illness": pd.to_numeric(data.get("avg_illness"), errors="coerce").lt(0),
        "non_positive_average_payment": pd.to_numeric(data.get("avg_Payment"), errors="coerce").le(0),
        "non_positive_average_workload": pd.to_numeric(data.get("avg_omes"), errors="coerce").le(0),
        "age_below_18": pd.to_numeric(data.get("age"), errors="coerce").lt(18),
        "negative_tenure": pd.to_numeric(data.get("vetek_months"), errors="coerce").lt(0),
    }
    invalid = pd.Series(False, index=data.index)
    for condition in conditions.values():
        invalid |= condition.fillna(False)

    by_source = {}
    if "source" in data.columns:
        for reason, condition in conditions.items():
            by_source[reason] = {
                str(source): int(count)
                for source, count in condition.fillna(False).groupby(data["source"]).sum().items()
            }

    retained_old_age = pd.to_numeric(data.get("age"), errors="coerce").gt(75)
    audit = CleaningAudit(
        input_rows=len(data),
        output_rows=int((~invalid).sum()),
        removed_rows=int(invalid.sum()),
        removed_positive_rows=int(data.loc[invalid, TARGET_COL].fillna(0).sum()),
        retained_age_above_75_rows=int((retained_old_age & ~invalid).sum()),
        reason_counts={name: int(mask.fillna(False).sum()) for name, mask in conditions.items()},
        reason_counts_by_source=by_source,
    )
    return data.loc[~invalid].copy(), audit


def _months_between(current: pd.Series, previous: pd.Series) -> pd.Series:
    return (current - previous).dt.total_seconds() / (30.4375 * 24 * 60 * 60)


def add_time_safe_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create current/prior history without using an employee's future rows."""
    data = frame.copy()
    if "calc_month" not in data.columns:
        raise ValueError("calc_month is required for chronological history features.")

    data["calc_month"] = pd.to_datetime(data["calc_month"], errors="coerce")
    data = data.sort_values(["source_employee_id", "calc_month"], kind="stable").copy()
    grouped = data.groupby("source_employee_id", sort=False)
    history_features: dict[str, pd.Series] = {"history_prior_records": grouped.cumcount()}

    first_date = grouped["calc_month"].transform("min")
    previous_date = grouped["calc_month"].shift(1)
    history_features["history_elapsed_months"] = _months_between(data["calc_month"], first_date).clip(lower=0)
    history_features["months_since_previous_record"] = _months_between(data["calc_month"], previous_date)
    months_since_previous = history_features["months_since_previous_record"]

    for column in HISTORY_NUMERIC_COLUMNS:
        if column not in data.columns:
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        data[column] = numeric
        group_values = data.groupby("source_employee_id", sort=False)[column]
        previous = group_values.shift(1)
        prior_mean = group_values.transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
        prior_std = group_values.transform(lambda values: values.shift(1).expanding(min_periods=2).std())
        delta = numeric - previous

        history_features[f"{column}_hist_prev"] = previous
        history_features[f"{column}_hist_delta_prev"] = delta
        history_features[f"{column}_hist_prior_mean"] = prior_mean
        history_features[f"{column}_hist_prior_std"] = prior_std
        history_features[f"{column}_hist_vs_prior_mean"] = numeric - prior_mean
        history_features[f"{column}_hist_slope_per_month"] = delta / months_since_previous.replace(0, np.nan)

    for column in HISTORY_CATEGORICAL_COLUMNS:
        if column not in data.columns:
            continue
        previous = data.groupby("source_employee_id", sort=False)[column].shift(1)
        current_text = data[column].astype("string").fillna("__missing__")
        previous_text = previous.astype("string").fillna("__missing__")
        history_features[f"{column}_hist_prev"] = previous
        history_features[f"{column}_hist_changed_from_prev"] = (
            previous.notna() & current_text.ne(previous_text)
        ).astype("int8")

    for column in PAYMENT_COLUMNS:
        if column not in data.columns:
            continue
        missing = pd.to_numeric(data[column], errors="coerce").isna().astype(float)
        history_features[f"{column}_missing_raw"] = missing.astype("int8")
        history_features[f"{column}_observed_fraction_to_date"] = (
            (1 - missing)
            .groupby(data["source_employee_id"], sort=False)
            .transform(lambda values: values.expanding().mean())
        )

    data = pd.concat([data, pd.DataFrame(history_features, index=data.index)], axis=1)
    return data.replace([np.inf, -np.inf], np.nan)


def add_eda_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if {"avg_Payment", "Median_Payment"}.issubset(data.columns):
        data["salary_mean_median_gap"] = data["avg_Payment"] - data["Median_Payment"]
    if {"stdevp_Payment", "avg_Payment"}.issubset(data.columns):
        data["salary_cv"] = data["stdevp_Payment"] / (data["avg_Payment"].abs() + 1)
    if {"salary_change", "avg_Payment"}.issubset(data.columns):
        data["salary_change_pct"] = data["salary_change"] / (data["avg_Payment"].abs() + 1)
    if {"stdevp_illness", "avg_illness"}.issubset(data.columns):
        data["illness_cv"] = data["stdevp_illness"] / (data["avg_illness"].abs() + 0.01)
    if {"stdevp_omes", "avg_omes"}.issubset(data.columns):
        data["workload_cv"] = data["stdevp_omes"] / (data["avg_omes"].abs() + 0.01)
    if {"age", "vetek_months"}.issubset(data.columns):
        data["career_start_age"] = data["age"] - data["vetek_months"] / 12
        data["tenure_years"] = data["vetek_months"] / 12
        data["is_new_employee"] = data["vetek_months"].lt(12).astype("int8")
    if {"count_managers", "history_prior_records"}.issubset(data.columns):
        data["manager_changes_per_prior_interval"] = (
            data["count_managers"] / data["history_prior_records"].clip(lower=1)
        )
    return data.replace([np.inf, -np.inf], np.nan)


def load_source_frames(paths: Mapping[str, str | Path] | None = None) -> pd.DataFrame:
    source_paths = paths or DEFAULT_SOURCE_PATHS
    frames = []
    for source, path in source_paths.items():
        frame, _ = read_excel_with_header_detection(path)
        frame = canonicalize_columns_and_categories(frame)
        if "target" in frame.columns:
            if not frame["target"].equals(frame[TARGET_COL]):
                raise ValueError(f"{source}: target differs from leave_ind")
            frame = frame.drop(columns="target")
        frame["source"] = source
        frame["source_employee_id"] = source + ":" + frame["fictive_employee"].astype(str)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    validate_target_metadata(combined)
    return combined


def model_feature_columns(
    frame: pd.DataFrame,
    *,
    excluded_columns: Iterable[str] = DEFAULT_EXCLUDED_MODEL_COLUMNS,
) -> tuple[list[str], list[str]]:
    blocked = set(excluded_columns) | OUTCOME_METADATA_COLUMNS | AUXILIARY_ONLY_COLUMNS | {TARGET_COL}
    selected = []
    dropped = []
    for column in frame.columns:
        lower = str(column).lower()
        is_outcome = "aziva" in lower or column in blocked
        is_date = pd.api.types.is_datetime64_any_dtype(frame[column])
        if is_outcome or is_date:
            dropped.append(column)
        else:
            selected.append(column)
    return selected, dropped


def prepare_turnover_data(
    paths: Mapping[str, str | Path] | None = None,
    *,
    excluded_model_columns: Iterable[str] = DEFAULT_EXCLUDED_MODEL_COLUMNS,
) -> PreparedTurnoverData:
    raw = load_source_frames(paths)
    clean, audit = clean_problematic_records(raw)
    history = add_time_safe_history_features(clean)
    history = add_eda_engineered_features(history)

    feature_columns, dropped = model_feature_columns(history, excluded_columns=excluded_model_columns)
    # Keep source/year as imputation context. They are explicitly excluded by
    # the final model ColumnTransformer.
    context_columns = [c for c in ["source", "year_date"] if c in history.columns]
    X_columns = list(dict.fromkeys(feature_columns + context_columns))

    train = history[history["source"].isin(TRAIN_SOURCES)].copy()
    test = history[history["source"].eq(EXTERNAL_TEST_SOURCE)].copy()
    if test.empty:
        raise ValueError("External test source file3 is empty after cleaning.")

    return PreparedTurnoverData(
        train_frame=train,
        test_frame=test,
        X_train=train[X_columns].copy(),
        y_train=train[TARGET_COL].astype(int).copy(),
        train_groups=train["source_employee_id"].astype(str).copy(),
        X_test=test[X_columns].copy(),
        y_test=test[TARGET_COL].astype(int).copy(),
        test_groups=test["source_employee_id"].astype(str).copy(),
        cleaning_audit=audit,
        dropped_model_columns=sorted(set(dropped)),
    )


def _make_model_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_fold_safe_model_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Return payment imputation + generic encoding, fitted within a fold."""
    frame = pd.DataFrame(X)
    model_columns = [
        c for c in frame.columns
        if c not in {"source", "year_date"}
        and not pd.api.types.is_datetime64_any_dtype(frame[c])
    ]
    numeric = [
        c for c in model_columns
        if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
    ]
    indicator_columns = [f"{column}_was_missing" for column in PAYMENT_COLUMNS]
    numeric = list(dict.fromkeys(numeric + indicator_columns))
    categorical = [c for c in model_columns if c not in numeric]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", _make_model_one_hot_encoder()),
    ])
    tabular = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("payment_imputer", SimilarEmployeePaymentImputer()),
        ("tabular", tabular),
    ])


def save_preprocessing_artifacts(
    prepared: PreparedTurnoverData,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Fit final file1+file2 preprocessing and save transform-only artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    preprocessor = make_fold_safe_model_preprocessor(prepared.X_train)
    preprocessor.fit(prepared.X_train, prepared.y_train)

    paths = {
        "preprocessor": output / "turnover_preprocessor.joblib",
        "train_frame": output / "train_employee_periods.pkl",
        "test_frame": output / "file3_employee_periods.pkl",
        "metadata": output / "preprocessing_metadata.json",
    }
    joblib.dump(preprocessor, paths["preprocessor"])
    prepared.train_frame.to_pickle(paths["train_frame"])
    prepared.test_frame.to_pickle(paths["test_frame"])
    metadata = {
        "train_rows": len(prepared.X_train),
        "test_rows": len(prepared.X_test),
        "train_employees": int(prepared.train_groups.nunique()),
        "test_employees": int(prepared.test_groups.nunique()),
        "train_positive_rate": float(prepared.y_train.mean()),
        "test_positive_rate": float(prepared.y_test.mean()),
        "cleaning_audit": asdict(prepared.cleaning_audit),
        "dropped_model_columns": prepared.dropped_model_columns,
        "note": "For internal validation, refit make_fold_safe_model_preprocessor inside each employee-grouped fold.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EDA-driven employee-period preprocessing artifacts.")
    parser.add_argument("--output-dir", default="output/preprocessed")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without saving artifacts.")
    args = parser.parse_args()

    prepared = prepare_turnover_data()
    print(json.dumps(asdict(prepared.cleaning_audit), indent=2))
    print(
        f"Train: {len(prepared.X_train):,} rows / {prepared.train_groups.nunique():,} employees / "
        f"{prepared.y_train.mean():.2%} positive"
    )
    print(
        f"Test:  {len(prepared.X_test):,} rows / {prepared.test_groups.nunique():,} employees / "
        f"{prepared.y_test.mean():.2%} positive"
    )
    print(f"Candidate model columns (plus imputation context): {prepared.X_train.shape[1]}")

    if not args.dry_run:
        paths = save_preprocessing_artifacts(prepared, args.output_dir)
        for name, path in paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()

