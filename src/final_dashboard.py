"""Dashboard helpers for the final EDA-driven turnover model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from src.preprocess import prepare_turnover_data


FINAL_ARTIFACT_PATH = Path("artifacts/final_best_model.pkl")


def risk_category(probability: float) -> str:
    if probability <= 0.3:
        return "Low Risk"
    if probability <= 0.5:
        return "Medium Risk"
    if probability <= 0.7:
        return "High Risk"
    return "Very High Risk"


def positive_probabilities(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    columns = artifact["feature_columns"]
    model = artifact["pipeline"]
    values = X.copy()
    for column in columns:
        if column not in values.columns:
            values[column] = pd.NA
    probabilities = model.predict_proba(values[columns])
    return np.asarray(probabilities[:, 1], dtype=float)


@lru_cache(maxsize=16)
def load_final_dashboard_bundle(artifact_path: str = str(FINAL_ARTIFACT_PATH)) -> dict:
    artifact = joblib.load(artifact_path)
    train_sources = artifact.get("train_sources", ["file1", "file2"])
    test_sources = artifact.get("test_sources", artifact.get("external_test_source", "file3"))
    prepared = prepare_turnover_data(train_sources=train_sources, test_sources=test_sources)
    columns = artifact["feature_columns"]

    raw_frame = prepared.test_frame.reset_index(drop=True).copy()
    model_frame = prepared.X_test.reset_index(drop=True).copy()
    probabilities = positive_probabilities(artifact, model_frame)

    app_frame = raw_frame.copy()
    app_frame["turnover_probability"] = probabilities
    app_frame["turnover_prediction"] = (probabilities >= float(artifact.get("decision_threshold", 0.5))).astype(int)
    app_frame["risk_rank"] = pd.Series(probabilities).rank(method="first", ascending=False).astype(int)
    app_frame["Risk Category"] = [risk_category(prob) for prob in probabilities]
    app_frame["Turnover Probability"] = probabilities
    app_frame["Employee ID"] = app_frame["fictive_employee"]
    app_frame["Budget Section"] = app_frame.get("Seif", pd.Series(index=app_frame.index, dtype=object))
    app_frame["Tenure (Months)"] = app_frame.get("vetek_months", pd.Series(index=app_frame.index, dtype=float))
    app_frame["Age"] = app_frame.get("age", pd.Series(index=app_frame.index, dtype=float))
    app_frame["Job Rank"] = app_frame.get("Maamad", pd.Series(index=app_frame.index, dtype=object))
    app_frame["Role Code"] = app_frame.get("tafkidCode", pd.Series(index=app_frame.index, dtype=object))
    app_frame["Contract Type"] = app_frame.get("contract_type", pd.Series(index=app_frame.index, dtype=object))
    app_frame["City"] = app_frame.get("Yishuv", pd.Series(index=app_frame.index, dtype=object))

    sort_cols = ["fictive_employee"]
    if "calc_month" in app_frame.columns:
        sort_cols.append("calc_month")
    dashboard_df = app_frame.sort_values(sort_cols).groupby("fictive_employee", as_index=False).tail(1).copy()
    dashboard_df = dashboard_df.sort_values("Turnover Probability", ascending=False).reset_index(drop=True)

    # The micro view needs model-ready history rows for what-if simulation.
    raw_for_app = raw_frame.copy()
    for column in model_frame.columns:
        if column not in raw_for_app.columns:
            raw_for_app[column] = model_frame[column]
    raw_for_app["Employee ID"] = raw_for_app["fictive_employee"]
    raw_for_app["Turnover Probability"] = probabilities
    raw_for_app["Risk Category"] = [risk_category(prob) for prob in probabilities]

    return {
        "artifact": artifact,
        "dashboard_df": dashboard_df,
        "raw_df": raw_for_app,
        "prepared": prepared,
        "metadata": {
            "dataset_label": (
                f"{artifact.get('candidate', 'final model')} - "
                f"{test_sources if isinstance(test_sources, str) else ', '.join(test_sources)} test"
            ),
            "artifact_path": artifact_path,
            "candidate": artifact.get("candidate", "final model"),
            "train_sources": train_sources,
            "test_sources": test_sources,
            "decision_threshold": float(artifact.get("decision_threshold", 0.5)),
        },
    }


def numeric_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
        return float(value) if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def set_if_present(frame: pd.DataFrame, index, column: str, value) -> None:
    if column in frame.columns:
        frame.at[index, column] = value


def update_delta_features(frame: pd.DataFrame, index, base_column: str, new_value: float) -> None:
    previous = numeric_value(frame.loc[index], f"{base_column}_hist_prev", np.nan)
    prior_mean = numeric_value(frame.loc[index], f"{base_column}_hist_prior_mean", np.nan)
    current = numeric_value(frame.loc[index], base_column, np.nan)

    set_if_present(frame, index, base_column, new_value)
    if pd.notna(previous):
        set_if_present(frame, index, f"{base_column}_hist_delta_prev", new_value - previous)
    if pd.notna(prior_mean):
        set_if_present(frame, index, f"{base_column}_hist_vs_prior_mean", new_value - prior_mean)
    if base_column == "avg_Payment" and "salary_change" in frame.columns:
        baseline_previous = previous if pd.notna(previous) else current
        if pd.notna(baseline_previous):
            set_if_present(frame, index, "salary_change", new_value - baseline_previous)
            set_if_present(frame, index, "salary_change_hist_delta_prev", new_value - baseline_previous)


def update_ratio_features(frame: pd.DataFrame, index) -> None:
    avg_payment = numeric_value(frame.loc[index], "avg_Payment", np.nan)
    avg_workload = numeric_value(frame.loc[index], "avg_omes", np.nan)
    avg_illness = numeric_value(frame.loc[index], "avg_illness", np.nan)
    stdev_workload = numeric_value(frame.loc[index], "stdevp_omes", np.nan)
    stdev_illness = numeric_value(frame.loc[index], "stdevp_illness", np.nan)
    stdev_payment = numeric_value(frame.loc[index], "stdevp_Payment", np.nan)

    if pd.notna(avg_workload) and abs(avg_workload) > 1e-9:
        set_if_present(frame, index, "workload_cv", stdev_workload / avg_workload if pd.notna(stdev_workload) else np.nan)
        set_if_present(frame, index, "workload_pay_ratio", avg_workload / max(avg_payment, 1.0) if pd.notna(avg_payment) else np.nan)
    if pd.notna(avg_illness) and abs(avg_illness) > 1e-9:
        set_if_present(frame, index, "illness_cv", stdev_illness / avg_illness if pd.notna(stdev_illness) else np.nan)
    if pd.notna(avg_payment) and abs(avg_payment) > 1e-9:
        set_if_present(frame, index, "salary_cv", stdev_payment / avg_payment if pd.notna(stdev_payment) else np.nan)


def apply_final_what_if(
    employee_rows: pd.DataFrame,
    *,
    salary: float | None = None,
    workload: float | None = None,
    illness: float | None = None,
    contract_type: object | None = None,
    maamad: object | None = None,
    seif: object | None = None,
) -> pd.DataFrame:
    """Apply final-model-aligned what-if edits to the latest employee row."""
    changed = employee_rows.copy()
    if changed.empty:
        return changed
    if "calc_month" in changed.columns:
        changed = changed.sort_values("calc_month")
    latest_idx = changed.index[-1]

    if salary is not None:
        update_delta_features(changed, latest_idx, "avg_Payment", float(salary))
    if workload is not None:
        update_delta_features(changed, latest_idx, "avg_omes", float(workload))
    if illness is not None:
        update_delta_features(changed, latest_idx, "avg_illness", float(illness))
    if contract_type is not None:
        set_if_present(changed, latest_idx, "contract_type", contract_type)
    if maamad is not None:
        set_if_present(changed, latest_idx, "Maamad", maamad)
    if seif is not None:
        set_if_present(changed, latest_idx, "Seif", seif)

    update_ratio_features(changed, latest_idx)
    return changed


def available_options(frame: pd.DataFrame, column: str, current) -> list:
    if column not in frame.columns:
        return [current] if pd.notna(current) else []
    values = frame[column].dropna().unique().tolist()
    if pd.notna(current) and current not in values:
        values.append(current)
    return sorted(values, key=lambda value: str(value))
