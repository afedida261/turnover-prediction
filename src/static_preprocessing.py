from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import TARGET_COL
from src.datasets import DatasetSpec


EXIT_METADATA_COLUMNS = ["aziva_kod", "aziva_date", "aziva_year", "target"]


def latest_employee_frame(raw_df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    data = raw_df.copy()
    if spec.employee_id_col not in data.columns:
        raise ValueError(f"Employee ID column '{spec.employee_id_col}' not found.")

    if spec.time_col and spec.time_col in data.columns:
        data = data.sort_values([spec.employee_id_col, spec.time_col])

    return data.groupby(spec.employee_id_col, as_index=False).last()


def add_static_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "avg_Payment" in data.columns and "Median_Payment" in data.columns:
        data["salary_skewness"] = (data["avg_Payment"] - data["Median_Payment"]) / (data["avg_Payment"].abs() + 1)

    if "stdevp_Payment" in data.columns and "avg_Payment" in data.columns:
        data["salary_cv"] = data["stdevp_Payment"] / (data["avg_Payment"].abs() + 1)

    if "change_in_salary_bySHKL" in data.columns and "avg_Payment" in data.columns:
        data["salary_change_pct"] = data["change_in_salary_bySHKL"] / (data["avg_Payment"].abs() + 1)

    if "stdevp_omes" in data.columns and "avg_omes" in data.columns:
        data["workload_stability"] = data["stdevp_omes"] / (data["avg_omes"].abs() + 0.01)

    if "avg_omes" in data.columns and "Median_omes" in data.columns:
        data["workload_skewness"] = (data["avg_omes"] - data["Median_omes"]) / (data["avg_omes"].abs() + 0.01)

    if "stdevp_illness" in data.columns and "avg_illness" in data.columns:
        data["illness_variability"] = data["stdevp_illness"] / (data["avg_illness"].abs() + 0.01)

    if "age" in data.columns and "vetek_months" in data.columns:
        data["career_start_age"] = data["age"] - (data["vetek_months"] / 12)
        denominator = ((data["age"] - 22).clip(lower=1) * 12)
        data["tenure_ratio"] = (data["vetek_months"] / denominator).clip(upper=1)

    if "vetek_months" in data.columns:
        data["tenure_years"] = data["vetek_months"] / 12
        data["is_new_employee"] = (data["vetek_months"] < 12).astype(int)
        data["is_senior"] = (data["vetek_months"] > 120).astype(int)

    if "age" in data.columns:
        data["is_young"] = (data["age"] < 30).astype(int)
        data["is_pre_retirement"] = (data["age"] > 55).astype(int)

    if "count_managers" in data.columns and "vetek_months" in data.columns:
        data["manager_change_rate"] = data["count_managers"] / (data["vetek_months"] / 12 + 0.1)

    if "avg_omes" in data.columns and "avg_Payment" in data.columns:
        data["workload_pay_ratio"] = data["avg_omes"] / (data["avg_Payment"].abs() + 1)

    if "distance_to_work" in data.columns:
        data["long_commute"] = (data["distance_to_work"] > 45).astype(int)

    return data.replace([np.inf, -np.inf], np.nan)


def build_static_model_frame(
    raw_df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    drop_exit_metadata: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    missing_cols = [col for col in [TARGET_COL, spec.employee_id_col] if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    dropped = [col for col in EXIT_METADATA_COLUMNS if drop_exit_metadata and col in raw_df.columns]
    model_frame = raw_df.drop(columns=dropped, errors="ignore").copy()
    model_frame = model_frame.replace([np.inf, -np.inf], np.nan)
    return model_frame, dropped


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def stringify_categorical_values(values):
    return pd.DataFrame(values).astype("string").fillna("__missing__").astype(str)


def make_tabular_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("stringify", FunctionTransformer(stringify_categorical_values, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []
