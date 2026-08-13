"""Employee-level Random Survival Forest experiment.

This module is intentionally separate from the final row-level classification
workflow. Files 1 and 2 form the development cohort and file 3 remains an
external test cohort. Each employee contributes exactly one baseline record.

Run the full experiment with::

    python -m src.analysis.random_survival_forest

Use ``--dry-run`` to build and audit the survival cohorts without fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest

from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

# Support both ``python -m src.analysis.random_survival_forest`` and direct
# execution via ``python src/analysis/random_survival_forest.py``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.survival_extensions import (
    DETERMINISTIC_REDUNDANT_COLUMNS,
    PooledLogisticSurvival,

    bootstrap_performance_intervals,
    calibration_table,
    save_calibration_plot,
    time_dependent_auc_metrics,
)
from src.imputations import PAYMENT_COLUMNS, SimilarEmployeePaymentImputer
from src.preprocess import PreparedTurnoverData, prepare_turnover_data


@contextmanager
def fitting_progress(label: str, *, update_seconds: float = 5.0):
    """Show an indeterminate activity bar while a blocking estimator fit runs."""
    if update_seconds <= 0:
        yield
        return

    started = time.monotonic()
    stopped = threading.Event()
    width = 20

    def report() -> None:
        tick = 0
        while not stopped.wait(update_seconds):
            cycle = 2 * width - 2
            position = tick % cycle
            if position >= width:
                position = cycle - position
            bar = ["-"] * width
            bar[position] = ">"
            elapsed = time.monotonic() - started
            print(
                f"[{label} fit] [{''.join(bar)}] elapsed {elapsed:,.0f}s",
                end="\r",
                flush=True,
            )
            tick += 1

    print(f"[{label} fit] started; elapsed-time activity updates every {update_seconds:g}s.")
    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    status = "completed"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        stopped.set()
        worker.join(timeout=max(update_seconds, 0.1) + 0.5)
        elapsed = time.monotonic() - started
        print(f"[{label} fit] {status} after {elapsed:,.1f}s." + " " * 30, flush=True)


SEED = 42
MONTH_DAYS = 365.25 / 12.0
DEFAULT_FOLLOW_UP_MONTHS = 12
MODEL_CONTEXT_COLUMNS = {"source", "year_date"}
OUTCOME_OR_ID_COLUMNS = {
    "leave_ind",
    "target",
    "aziva_kod",
    "aziva_date",
    "aziva_year",
    "source_employee_id",
    "fictive_employee",
    "calc_month",
}
PREFERRED_BRIER_HORIZONS = (6.0, 12.0, 24.0, 36.0)

DEFAULT_RSF_PARAMETERS: dict[str, object] = {
    "min_samples_leaf": 25,
    "min_samples_split": 50,
    "max_features": 0.5,
    "max_depth": None,
    "max_samples": None,
}
RSF_TUNING_SPACE: dict[str, tuple[object, ...]] = {
    "min_samples_leaf": (25, 50, 100, 150),
    "max_features": ("sqrt", 0.15, 0.30, 0.50),
    "max_depth": (8, 12, None),
    "max_samples": (0.60, 0.80, None),
}
BEST_PARAMETERS_FILENAME = "best_hyperparameters.json"

@dataclass(frozen=True)
class CohortAudit:
    source_names: tuple[str, ...]
    input_rows: int
    input_employees: int
    output_employees: int
    events: int
    censored: int
    multiple_event_employees: int
    post_event_rows_ignored: int
    missing_baseline_date_employees: int
    missing_event_date_employees: int
    nonpositive_followup_employees: int
    censor_follow_up_months: int


@dataclass
class SurvivalCohort:
    X: pd.DataFrame
    y: np.ndarray
    metadata: pd.DataFrame
    audit: CohortAudit


def _months_between(end: pd.Timestamp, start: pd.Timestamp) -> float:
    return float((end - start).total_seconds() / (MONTH_DAYS * 24 * 60 * 60))


def _survival_target(events: Sequence[bool], times: Sequence[float]) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(events, dtype=bool),
        time=np.asarray(times, dtype=float),
        name_event="event",
        name_time="time",
    )


def build_employee_survival_cohort(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    censor_follow_up_months: int = DEFAULT_FOLLOW_UP_MONTHS,
    strict: bool = True,
) -> SurvivalCohort:
    """Collapse employee-period rows into one right-censored record per employee.

    Covariates come only from the earliest available row. For employees who
    leave, observed time ends at their first valid ``aziva_date``. Employees
    without an event are known to survive through 12 months after their final
    ``calc_month`` and are censored there.
    """
    required = {"source_employee_id", "source", "calc_month", "leave_ind", "aziva_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Survival cohort is missing required columns: {sorted(missing)}")
    if censor_follow_up_months <= 0:
        raise ValueError("censor_follow_up_months must be positive.")

    features = list(dict.fromkeys(feature_columns))
    forbidden = sorted(set(features).intersection(OUTCOME_OR_ID_COLUMNS))
    if forbidden:
        raise ValueError(f"Outcome, identifier, or time columns cannot be covariates: {forbidden}")
    unavailable = sorted(set(features).difference(frame.columns))
    if unavailable:
        raise ValueError(f"Requested feature columns are missing: {unavailable}")

    data = frame.copy()
    data["calc_month"] = pd.to_datetime(data["calc_month"], errors="coerce")
    data["aziva_date"] = pd.to_datetime(data["aziva_date"], errors="coerce")
    data["leave_ind"] = pd.to_numeric(data["leave_ind"], errors="coerce")
    data = data.sort_values(["source_employee_id", "calc_month"], kind="stable")

    records: list[dict[str, object]] = []
    feature_rows: list[pd.Series] = []
    missing_baseline = 0
    missing_event_date = 0
    nonpositive_followup = 0
    multiple_event_employees = 0
    post_event_rows_ignored = 0

    for employee_id, employee_rows in data.groupby("source_employee_id", sort=False):
        employee_rows = employee_rows.sort_values("calc_month", kind="stable")
        valid_dates = employee_rows["calc_month"].notna()
        if not valid_dates.any():
            missing_baseline += 1
            continue

        employee_rows = employee_rows.loc[valid_dates].copy()
        baseline_row = employee_rows.iloc[0]
        baseline_date = pd.Timestamp(baseline_row["calc_month"])
        event_rows = employee_rows.loc[employee_rows["leave_ind"].eq(1)].copy()
        event_count = len(event_rows)
        multiple_event_employees += int(event_count > 1)

        if event_count:
            valid_event_rows = event_rows.loc[event_rows["aziva_date"].notna()].sort_values("aziva_date")
            if valid_event_rows.empty:
                missing_event_date += 1
                continue
            first_event_row = valid_event_rows.iloc[0]
            endpoint_date = pd.Timestamp(first_event_row["aziva_date"])
            first_event_position = int(employee_rows.index.get_loc(first_event_row.name))
            post_event_rows_ignored += max(len(employee_rows) - first_event_position - 1, 0)
            event = True
            event_date: pd.Timestamp | pd.NaT = endpoint_date
        else:
            last_observed_date = pd.Timestamp(employee_rows["calc_month"].max())
            endpoint_date = last_observed_date + pd.DateOffset(months=censor_follow_up_months)
            event = False
            event_date = pd.NaT

        observed_time = _months_between(endpoint_date, baseline_date)
        if not np.isfinite(observed_time) or observed_time <= 0:
            nonpositive_followup += 1
            continue

        last_observed_date = pd.Timestamp(employee_rows["calc_month"].max())
        feature_rows.append(baseline_row.loc[features].copy())
        records.append(
            {
                "source_employee_id": str(employee_id),
                "source": str(baseline_row["source"]),
                "baseline_date": baseline_date,
                "last_observed_date": last_observed_date,
                "event_date": event_date,
                "endpoint_date": endpoint_date,
                "observed_time_months": observed_time,
                "event": event,
                "employee_period_rows": int(len(employee_rows)),
            }
        )

    invalid_counts = {
        "missing baseline dates": missing_baseline,
        "missing event dates": missing_event_date,
        "non-positive follow-up": nonpositive_followup,
    }
    invalid_counts = {name: count for name, count in invalid_counts.items() if count}
    if strict and invalid_counts:
        details = ", ".join(f"{name}={count}" for name, count in invalid_counts.items())
        raise ValueError(f"Invalid employee survival records: {details}")
    if not records:
        raise ValueError("No valid employees remain in the survival cohort.")

    metadata = pd.DataFrame(records).reset_index(drop=True)
    X = pd.DataFrame(feature_rows).reset_index(drop=True)
    if len(X) != len(metadata):  # defensive alignment check
        raise RuntimeError("Survival features and outcomes are misaligned.")
    if metadata["source_employee_id"].duplicated().any():
        raise RuntimeError("Survival cohort contains duplicate employees.")

    y = _survival_target(metadata["event"], metadata["observed_time_months"])
    sources = tuple(sorted(metadata["source"].dropna().astype(str).unique()))
    audit = CohortAudit(
        source_names=sources,
        input_rows=int(len(data)),
        input_employees=int(data["source_employee_id"].nunique()),
        output_employees=int(len(metadata)),
        events=int(metadata["event"].sum()),
        censored=int((~metadata["event"]).sum()),
        multiple_event_employees=multiple_event_employees,
        post_event_rows_ignored=post_event_rows_ignored,
        missing_baseline_date_employees=missing_baseline,
        missing_event_date_employees=missing_event_date,
        nonpositive_followup_employees=nonpositive_followup,
        censor_follow_up_months=int(censor_follow_up_months),
    )
    return SurvivalCohort(X=X, y=y, metadata=metadata, audit=audit)


def make_employee_survival_cohorts(
    prepared: PreparedTurnoverData,
    *,
    censor_follow_up_months: int = DEFAULT_FOLLOW_UP_MONTHS,
) -> tuple[SurvivalCohort, SurvivalCohort]:
    """Build aligned development and external cohorts from prepared data."""
    feature_columns = list(prepared.X_train.columns)
    train = build_employee_survival_cohort(
        prepared.train_frame,
        feature_columns=feature_columns,
        censor_follow_up_months=censor_follow_up_months,
    )
    test = build_employee_survival_cohort(
        prepared.test_frame,
        feature_columns=feature_columns,
        censor_follow_up_months=censor_follow_up_months,
    )
    return train, test


def remove_uninformative_training_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop deterministic redundancies and covariates with no training variation."""
    dropped = []
    for column in X_train.columns:
        if column in MODEL_CONTEXT_COLUMNS:
            continue
        if column in DETERMINISTIC_REDUNDANT_COLUMNS or X_train[column].nunique(dropna=False) <= 1:
            dropped.append(column)
    kept = [column for column in X_train.columns if column not in dropped]
    return X_train.loc[:, kept].copy(), X_test.loc[:, kept].copy(), dropped


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - old sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _as_dense_matrix(X):
    """Use dense float32 data to avoid scikit-survival 0.25's sparse OOB bug."""
    values = X.toarray() if sparse.issparse(X) else X
    return np.asarray(values, dtype=np.float32, order="C")


def make_rsf_preprocessor(X: pd.DataFrame, *, seed: int = SEED) -> Pipeline:
    """Create train-fitted payment imputation and RSF tabular preprocessing."""
    model_columns = [
        column
        for column in X.columns
        if column not in MODEL_CONTEXT_COLUMNS
        and not pd.api.types.is_datetime64_any_dtype(X[column])
    ]
    numeric = [
        column
        for column in model_columns
        if pd.api.types.is_numeric_dtype(X[column]) or pd.api.types.is_bool_dtype(X[column])
    ]
    payment_indicators = [
        f"{column}_was_missing" for column in PAYMENT_COLUMNS if column in X.columns
    ]
    numeric = list(dict.fromkeys(numeric + payment_indicators))
    categorical = [column for column in model_columns if column not in numeric]

    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable RSF covariates remain after exclusions.")

    tabular = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("payment_imputer", SimilarEmployeePaymentImputer(random_state=seed)),
            ("tabular", tabular),
            (
                "to_dense",
                FunctionTransformer(_as_dense_matrix, accept_sparse=True, validate=False),
            ),
        ]
    )


def _ipcw_concordance(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    risk: np.ndarray,
    *,
    tau: float,
) -> float:
    return float(concordance_index_ipcw(survival_train, survival_test, risk, tau=tau)[0])


def _conservative_tau(y_train: np.ndarray, y_test: np.ndarray | None = None) -> float:
    train_times = np.asarray(y_train["time"], dtype=float)
    arrays = [train_times]
    if y_test is not None:
        arrays.append(np.asarray(y_test["time"], dtype=float))
    upper = min(float(np.quantile(values, 0.80)) for values in arrays)
    absolute_upper = min(float(np.max(values)) for values in arrays)
    tau = min(upper, np.nextafter(absolute_upper, -np.inf))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Could not determine a positive IPCW evaluation horizon.")
    return tau


def _new_rsf(
    *,
    params: dict[str, object],
    n_estimators: int,
    seed: int,
    n_jobs: int,
    warm_start: bool = False,
) -> RandomSurvivalForest:
    return RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=params["max_features"],
        max_depth=params.get("max_depth"),
        max_samples=params.get("max_samples"),
        bootstrap=True,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=seed,
        warm_start=warm_start,
        low_memory=False,
    )


def _normalize_rsf_parameters(parameters: dict[str, object]) -> dict[str, object]:
    required = {"min_samples_leaf", "max_features", "max_depth", "max_samples"}
    missing = required.difference(parameters)
    if missing:
        raise ValueError(f"Saved RSF parameters are missing: {sorted(missing)}")
    leaf = int(parameters["min_samples_leaf"])
    if leaf < 1:
        raise ValueError("min_samples_leaf must be positive.")
    return {
        "min_samples_leaf": leaf,
        "min_samples_split": 2 * leaf,
        "max_features": parameters["max_features"],
        "max_depth": None if parameters["max_depth"] is None else int(parameters["max_depth"]),
        "max_samples": None if parameters["max_samples"] is None else float(parameters["max_samples"]),
    }


def load_rsf_parameters(path: Path) -> tuple[dict[str, object], str]:
    """Load a prior tuning winner, otherwise use the documented default winner."""
    if not path.exists():
        return dict(DEFAULT_RSF_PARAMETERS), "documented_default"
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("selected_parameters", payload)
    if not isinstance(parameters, dict):
        raise ValueError(f"Invalid RSF parameter file: {path}")
    return _normalize_rsf_parameters(parameters), str(path)


def build_rsf_tuning_grid(*, max_candidates: int = 24, seed: int = SEED) -> list[dict[str, object]]:
    """Build a reproducible subset of the four-parameter Cartesian search space."""
    candidates = [
        _normalize_rsf_parameters(
            {
                "min_samples_leaf": leaf,
                "max_features": features,
                "max_depth": depth,
                "max_samples": samples,
            }
        )
        for leaf, features, depth, samples in product(
            RSF_TUNING_SPACE["min_samples_leaf"],
            RSF_TUNING_SPACE["max_features"],
            RSF_TUNING_SPACE["max_depth"],
            RSF_TUNING_SPACE["max_samples"],
        )
    ]
    default = _normalize_rsf_parameters(DEFAULT_RSF_PARAMETERS)
    candidates = [candidate for candidate in candidates if candidate != default]
    if max_candidates <= 0 or max_candidates >= len(candidates) + 1:
        return [default, *candidates]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=max_candidates - 1, replace=False)
    return [default, *(candidates[int(index)] for index in chosen)]


def tune_rsf_factory_transfer(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    parameter_grid: Sequence[dict[str, object]],
    n_estimators: int = 250,
    seed: int = SEED,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select RSF parameters by bidirectional factory-transfer performance."""
    if n_estimators < 10:
        raise ValueError("At least 10 tuning trees are required.")
    if "source" not in X.columns:
        raise ValueError("Factory-transfer tuning requires the source column.")
    sources = sorted(X["source"].dropna().astype(str).unique().tolist())
    if len(sources) != 2:
        raise ValueError(f"Expected exactly two development factories, found {sources}.")

    transfers = []
    source_values = X["source"].astype(str).to_numpy()
    for validation_source in sources:
        validation_indices = np.flatnonzero(source_values == validation_source)
        fit_indices = np.flatnonzero(source_values != validation_source)
        preprocessor = make_rsf_preprocessor(X.iloc[fit_indices], seed=seed)
        X_fit = preprocessor.fit_transform(X.iloc[fit_indices], y[fit_indices])
        X_validation = preprocessor.transform(X.iloc[validation_indices])
        transfers.append(
            (
                sources[0] if validation_source == sources[1] else sources[1],
                validation_source,
                X_fit,
                y[fit_indices],
                X_validation,
                y[validation_indices],
            )
        )

    rows = []
    total = len(parameter_grid)
    for candidate_number, raw_parameters in enumerate(parameter_grid, start=1):
        parameters = _normalize_rsf_parameters(dict(raw_parameters))
        transfer_ibs = []
        transfer_uno = []
        transfer_harrell = []
        row: dict[str, object] = {
            "Candidate": candidate_number,
            "Trees": n_estimators,
            "Min_Samples_Leaf": parameters["min_samples_leaf"],
            "Min_Samples_Split": parameters["min_samples_split"],
            "Max_Features": parameters["max_features"],
            "Max_Depth": parameters["max_depth"],
            "Max_Samples": parameters["max_samples"],
        }
        for direction, (
            fit_source,
            validation_source,
            X_fit,
            y_fit,
            X_validation,
            y_validation,
        ) in enumerate(transfers, start=1):
            model = _new_rsf(
                params=parameters,
                n_estimators=n_estimators,
                seed=seed + direction,
                n_jobs=n_jobs,
            )
            model.set_params(oob_score=False)
            progress_label = (
                f"RSF tuning {candidate_number}/{total} "
                f"{fit_source}->{validation_source}"
            )
            with fitting_progress(progress_label, update_seconds=5.0):
                model.fit(X_fit, y_fit)
            metrics, _, _, _ = evaluate_external_test(model, X_validation, y_fit, y_validation)
            prefix = f"{fit_source}_to_{validation_source}"
            row[f"{prefix}_IBS"] = metrics["Test_IBS"]
            row[f"{prefix}_Uno_C"] = metrics["Test_Uno_IPCW_C"]
            row[f"{prefix}_Harrell_C"] = metrics["Test_Harrell_C"]
            transfer_ibs.append(metrics["Test_IBS"])
            transfer_uno.append(metrics["Test_Uno_IPCW_C"])
            transfer_harrell.append(metrics["Test_Harrell_C"])
        row["Mean_Transfer_IBS"] = float(np.mean(transfer_ibs))
        row["Mean_Transfer_Uno_C"] = float(np.mean(transfer_uno))
        row["Mean_Transfer_Harrell_C"] = float(np.mean(transfer_harrell))
        rows.append(row)
        print(
            f"[RSF tuning] candidate {candidate_number}/{total} ({candidate_number / total:.0%}) "
            f"mean IBS={row['Mean_Transfer_IBS']:.5f}, Uno C={row['Mean_Transfer_Uno_C']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(
        ["Mean_Transfer_IBS", "Mean_Transfer_Uno_C"],
        ascending=[True, False],
    ).reset_index(drop=True)
    best = results.iloc[0]
    best_parameters = _normalize_rsf_parameters(
        {
            "min_samples_leaf": best["Min_Samples_Leaf"],
            "max_features": best["Max_Features"],
            "max_depth": None if pd.isna(best["Max_Depth"]) else best["Max_Depth"],
            "max_samples": None if pd.isna(best["Max_Samples"]) else best["Max_Samples"],
        }
    )
    return results, best_parameters

def fit_final_rsf_with_stability(
    X_train,
    y_train: np.ndarray,
    *,
    params: dict[str, object],
    tree_counts: Sequence[int],
    seed: int = SEED,
    n_jobs: int = -1,
) -> tuple[RandomSurvivalForest, pd.DataFrame]:
    """Grow the selected forest incrementally and record OOB stabilization."""
    counts = sorted(set(int(value) for value in tree_counts if int(value) > 0))
    if not counts:
        raise ValueError("At least one positive tree count is required.")
    X_train = _as_dense_matrix(X_train)
    tau = _conservative_tau(y_train)
    model = _new_rsf(
        params=params,
        n_estimators=counts[0],
        seed=seed,
        n_jobs=n_jobs,
        warm_start=True,
    )
    rows = []
    for count in counts:
        model.set_params(n_estimators=count)
        model.fit(X_train, y_train)
        rows.append(
            {
                "Trees": count,
                "OOB_Harrell_C": float(model.oob_score_),
                "OOB_Uno_C": _ipcw_concordance(
                    y_train,
                    y_train,
                    np.asarray(model.oob_prediction_, dtype=float),
                    tau=tau,
                ),
                "IPCW_Tau_Months": tau,
            }
        )
    return model, pd.DataFrame(rows)


def choose_evaluation_times(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    preferred_horizons: Sequence[float] = PREFERRED_BRIER_HORIZONS,
    grid_points: int = 60,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Choose Brier/IBS times inside conservative common follow-up support."""
    train_times = np.asarray(y_train["time"], dtype=float)
    test_times = np.asarray(y_test["time"], dtype=float)
    lower = max(float(np.min(test_times)), 1.0)
    upper = _conservative_tau(y_train, y_test)
    upper = float(np.nextafter(upper, -np.inf))
    if upper <= lower:
        raise ValueError("Common train/test follow-up is too narrow for Brier evaluation.")
    count = max(int(grid_points), 2)
    grid = np.linspace(lower, upper, num=count, endpoint=True, dtype=float)
    reported = np.asarray(
        [float(value) for value in preferred_horizons if lower <= float(value) <= upper],
        dtype=float,
    )
    if reported.size == 0:
        reported = np.quantile(grid, [0.25, 0.50, 0.75])
    return grid, reported, upper


def administratively_truncate_survival(y: np.ndarray, *, tau: float) -> np.ndarray:
    """Right-censor follow-up beyond ``tau`` without changing earlier events."""
    if tau <= 0:
        raise ValueError("Administrative truncation time must be positive.")
    original_times = np.asarray(y["time"], dtype=float)
    times = np.minimum(original_times, float(tau))
    events = np.asarray(y["event"], dtype=bool) & (original_times <= tau)
    return _survival_target(events, times)


def _brier_evaluation_target(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Keep external follow-up strictly inside the training censoring support."""
    train_max = float(np.max(y_train["time"]))
    if float(np.max(y_test["time"])) < train_max:
        return y_test
    cap = float(np.nextafter(train_max, -np.inf))
    return administratively_truncate_survival(y_test, tau=cap)


def _survival_probability_matrix(estimator, X, times: np.ndarray) -> np.ndarray:
    functions = estimator.predict_survival_function(X)
    return np.vstack([function(times) for function in functions])


def _kaplan_meier_probabilities(y_train: np.ndarray, times: np.ndarray, n_rows: int) -> np.ndarray:
    km_times, km_probabilities = kaplan_meier_estimator(y_train["event"], y_train["time"])
    positions = np.searchsorted(km_times, times, side="right") - 1
    values = np.ones(len(times), dtype=float)
    valid = positions >= 0
    values[valid] = km_probabilities[positions[valid]]
    return np.tile(values, (n_rows, 1))


def evaluate_external_test(
    estimator,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute external discrimination and calibration metrics."""
    grid, reported_times, tau = choose_evaluation_times(y_train, y_test)
    y_test_brier = _brier_evaluation_target(y_train, y_test)
    risk = np.asarray(estimator.predict(X_test), dtype=float)

    survival_grid = _survival_probability_matrix(estimator, X_test, grid)
    survival_reported = _survival_probability_matrix(estimator, X_test, reported_times)

    harrell = float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
    uno = _ipcw_concordance(y_train, y_test, risk, tau=tau)
    _, rsf_brier = brier_score(y_train, y_test_brier, survival_grid, grid)
    rsf_ibs = float(integrated_brier_score(y_train, y_test_brier, survival_grid, grid))

    km_grid = _kaplan_meier_probabilities(y_train, grid, len(y_test))
    _, km_brier = brier_score(y_train, y_test_brier, km_grid, grid)
    km_ibs = float(integrated_brier_score(y_train, y_test_brier, km_grid, grid))

    report_probability = _survival_probability_matrix(estimator, X_test, reported_times)
    _, report_brier = brier_score(y_train, y_test_brier, report_probability, reported_times)
    metrics: dict[str, float] = {
        "Test_Harrell_C": harrell,
        "Test_Uno_IPCW_C": uno,
        "IPCW_Tau_Months": tau,
        "Test_IBS": rsf_ibs,
        "Kaplan_Meier_IBS": km_ibs,
        "IBS_Improvement_Over_KM": km_ibs - rsf_ibs,
    }
    for time, value in zip(reported_times, report_brier):
        metrics[f"Test_Brier_{time:g}_Months"] = float(value)

    curve = pd.DataFrame(
        {
            "Time_Months": grid,
            "RSF_Brier": rsf_brier,
            "Kaplan_Meier_Brier": km_brier,
        }
    )
    return metrics, curve, risk, survival_reported


def write_extended_performance_outputs(
    estimator,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    output: Path,
    slug: str,
    label: str,
    bootstrap_repeats: int,
    calibration_bins: int,
    seed: int,
) -> dict[str, float]:
    """Write dynamic AUC, calibration, and external bootstrap outputs."""
    grid, horizons, tau = choose_evaluation_times(y_train, y_test)
    y_evaluation = _brier_evaluation_target(y_train, y_test)
    risk = np.asarray(estimator.predict(X_test), dtype=float)
    if np.isfinite(risk).all() and np.unique(risk).size < 2:
        print(
            f"WARNING: {label} produced a constant external-test risk score; "
            "discrimination is 0.5 and calibration will use one aggregate bin.",
            flush=True,
        )
    survival_grid = _survival_probability_matrix(estimator, X_test, grid)
    survival_horizons = _survival_probability_matrix(estimator, X_test, horizons)

    auc_metrics, auc_table = time_dependent_auc_metrics(
        y_train,
        y_evaluation,
        survival_horizons,
        horizons,
    )
    auc_table.to_csv(output / f"{slug}_time_dependent_auc.csv", index=False)
    _save_line_plot(
        auc_table,
        "Time_Months",
        ["Dynamic_AUC"],
        output / f"{slug}_time_dependent_auc.png",
        "Cumulative/dynamic AUC",
    )

    calibration = calibration_table(
        y_evaluation,
        survival_horizons,
        horizons,
        n_bins=calibration_bins,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    calibration.to_csv(output / f"{slug}_calibration.csv", index=False)
    save_calibration_plot(
        calibration,
        output / f"{slug}_calibration.png",
        title=f"{label} external calibration",
    )

    intervals = bootstrap_performance_intervals(
        y_train,
        y_evaluation,
        risk,
        survival_grid,
        grid,
        survival_horizons,
        horizons,
        tau=tau,
        repeats=bootstrap_repeats,
        seed=seed,
        progress_label=label,
    )
    intervals.to_csv(output / f"{slug}_bootstrap_cis.csv", index=False)
    return auc_metrics

def permutation_vimp(
    estimator,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    repeats: int = 20,
    seed: int = SEED,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute C-index and IBS VIMP together with visible progress."""
    if repeats < 1:
        raise ValueError("Permutation repeats must be positive.")
    if n_jobs != 1:
        print("Progress-aware VIMP runs sequentially; --permutation-jobs is ignored.", flush=True)

    grid, _, tau = choose_evaluation_times(y_train, y_test)
    y_evaluation = _brier_evaluation_target(y_train, y_test)
    baseline_risk = np.asarray(estimator.predict(X_test), dtype=float)
    baseline_survival = _survival_probability_matrix(estimator, X_test, grid)
    baseline_c = _ipcw_concordance(y_train, y_test, baseline_risk, tau=tau)
    baseline_ibs = float(integrated_brier_score(y_train, y_evaluation, baseline_survival, grid))

    features = [column for column in X_test.columns if column not in MODEL_CONTEXT_COLUMNS]
    c_importances = np.empty((len(features), repeats), dtype=float)
    ibs_importances = np.empty((len(features), repeats), dtype=float)
    total = len(features) * repeats
    completed = 0
    rng = np.random.default_rng(seed)

    for feature_number, feature in enumerate(features, start=1):
        permuted = X_test.copy()
        original = X_test[feature].to_numpy(copy=True)
        for repeat in range(repeats):
            permuted[feature] = rng.permutation(original)
            permuted_risk = np.asarray(estimator.predict(permuted), dtype=float)
            permuted_survival = _survival_probability_matrix(estimator, permuted, grid)
            permuted_c = _ipcw_concordance(y_train, y_test, permuted_risk, tau=tau)
            permuted_ibs = float(
                integrated_brier_score(y_train, y_evaluation, permuted_survival, grid)
            )
            c_importances[feature_number - 1, repeat] = baseline_c - permuted_c
            ibs_importances[feature_number - 1, repeat] = permuted_ibs - baseline_ibs
            completed += 1
            print(
                f"[VIMP] {completed}/{total} ({completed / total:.1%}) | "
                f"feature {feature_number}/{len(features)}: {feature} | repeat {repeat + 1}/{repeats}",
                end="\r",
                flush=True,
            )
        print(
            f"[VIMP] completed feature {feature_number}/{len(features)}: {feature}" + " " * 30,
            flush=True,
        )

    c_frame = pd.DataFrame(
        {
            "Feature": features,
            "C_Index_Decrease_Mean": c_importances.mean(axis=1),
            "C_Index_Decrease_Std": c_importances.std(axis=1, ddof=1) if repeats > 1 else 0.0,
        }
    ).sort_values("C_Index_Decrease_Mean", ascending=False, ignore_index=True)
    ibs_frame = pd.DataFrame(
        {
            "Feature": features,
            "IBS_Increase_Mean": ibs_importances.mean(axis=1),
            "IBS_Increase_Std": ibs_importances.std(axis=1, ddof=1) if repeats > 1 else 0.0,
        }
    ).sort_values("IBS_Increase_Mean", ascending=False, ignore_index=True)
    return c_frame, ibs_frame

def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def _save_line_plot(frame: pd.DataFrame, x: str, ys: Sequence[str], path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in ys:
        ax.plot(frame[x], frame[column], marker="o" if len(frame) < 10 else None, label=column)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_vimp_plot(frame: pd.DataFrame, value_column: str, error_column: str, path: Path) -> None:
    display = frame.head(25).sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(display))))
    ax.barh(display["Feature"], display[value_column], xerr=display[error_column], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(value_column.replace("_", " "))
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cohort_summary(cohort: SurvivalCohort, split: str) -> dict[str, object]:
    times = cohort.metadata["observed_time_months"]
    return {
        "Split": split,
        "Sources": ",".join(cohort.audit.source_names),
        "Employees": len(cohort.metadata),
        "Events": int(cohort.metadata["event"].sum()),
        "Censored": int((~cohort.metadata["event"]).sum()),
        "Event_Rate": float(cohort.metadata["event"].mean()),
        "Followup_Min_Months": float(times.min()),
        "Followup_Median_Months": float(times.median()),
        "Followup_Max_Months": float(times.max()),
    }


def _parse_tree_counts(text: str, final_trees: int) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    values.append(int(final_trees))
    values = sorted(set(value for value in values if 0 < value <= final_trees))
    if not values:
        raise ValueError("No valid stability tree counts were provided.")
    return values


def _write_readme(
    path: Path,
    *,
    train: SurvivalCohort,
    test: SurvivalCohort,
    metrics: dict[str, float] | None,
    best_params: dict[str, object] | None,
) -> None:
    lines = [
        "RANDOM SURVIVAL FOREST",
        "=" * 80,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Design",
        "- One baseline row per employee; baseline is the earliest valid calc_month.",
        "- Event time is exact aziva_date minus baseline date.",
        "- Non-events are censored 12 months after their last calc_month.",
        "- Files 1+2 are development data; file 3 is external test only.",
        "- OOB concordance is used for internal RSF selection.",
        "",
        "Cohorts",
        f"- Train: {len(train.metadata):,} employees / {int(train.metadata['event'].sum()):,} events",
        f"- Test: {len(test.metadata):,} employees / {int(test.metadata['event'].sum()):,} events",
    ]
    if best_params is not None:
        lines.extend(["", "Selected parameters", json.dumps(_json_ready(best_params), indent=2)])
    if metrics is not None:
        lines.extend(["", "External test metrics", json.dumps(_json_ready(metrics), indent=2)])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_rsf_experiment(
    *,
    output_dir: str | Path = "output/rsf",
    artifact_dir: str | Path = "artifacts",
    censor_follow_up_months: int = DEFAULT_FOLLOW_UP_MONTHS,
    tuning_trees: int = 250,
    final_trees: int = 750,
    stability_tree_counts: str = "100,250,500",
    permutation_repeats: int = 20,
    bootstrap_repeats: int = 500,
    calibration_bins: int = 5,


    discrete_interval_months: float = 6.0,
    discrete_C: float = 1.0,
    discrete_max_iterations: int = 2000,
    fit_progress_seconds: float = 5.0,
    seed: int = SEED,
    n_jobs: int = -1,
    permutation_jobs: int = 1,
    tune_hyperparameters: bool = False,
    tuning_max_candidates: int = 24,
    skip_permutation: bool = False,
    skip_baselines: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Refit RSF and optional pooled discrete-time model; evaluate on untouched file 3."""
    output = Path(output_dir)
    artifacts = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_configuration = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "censor_follow_up_months": censor_follow_up_months,
        "tuning_trees": tuning_trees,
        "final_trees": final_trees,
        "stability_tree_counts": stability_tree_counts,
        "permutation_repeats": permutation_repeats,
        "bootstrap_repeats": bootstrap_repeats,
        "calibration_bins": calibration_bins,



        "discrete_interval_months": discrete_interval_months,
        "discrete_C": discrete_C,
        "discrete_max_iterations": discrete_max_iterations,
        "fit_progress_seconds": fit_progress_seconds,
        "seed": seed,
        "n_jobs": n_jobs,
        "tune_hyperparameters": tune_hyperparameters,
        "tuning_max_candidates": tuning_max_candidates,
        "skip_vimp": skip_permutation,
        "skip_baselines": skip_baselines,
        "dry_run": dry_run,
    }
    (output / "run_configuration.json").write_text(
        json.dumps(_json_ready(run_configuration), indent=2), encoding="utf-8"
    )

    prepared = prepare_turnover_data()
    train, test = make_employee_survival_cohorts(
        prepared,
        censor_follow_up_months=censor_follow_up_months,
    )
    train.X, test.X, dropped_features = remove_uninformative_training_features(train.X, test.X)
    deterministic_dropped = sorted(DETERMINISTIC_REDUNDANT_COLUMNS.intersection(dropped_features))
    constant_dropped = sorted(set(dropped_features).difference(deterministic_dropped))

    cohort_audit = {
        "train": asdict(train.audit),
        "external_test": asdict(test.audit),
        "cleaning": asdict(prepared.cleaning_audit),
        "deterministic_redundant_features_dropped": deterministic_dropped,
        "constant_or_empty_features_dropped": constant_dropped,
    }
    (output / "cohort_audit.json").write_text(
        json.dumps(_json_ready(cohort_audit), indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [_cohort_summary(train, "file1+file2_train"), _cohort_summary(test, "file3_external_test")]
    ).to_csv(output / "cohort_summary.csv", index=False)
    train.metadata.to_csv(output / "train_survival_cohort.csv", index=False)
    test.metadata.to_csv(output / "file3_survival_cohort.csv", index=False)

    print(
        f"Train cohort: {len(train.metadata):,} employees / {int(train.metadata['event'].sum()):,} events"
    )
    print(
        f"Test cohort:  {len(test.metadata):,} employees / {int(test.metadata['event'].sum()):,} events"
    )
    print(f"Deterministic redundant features dropped: {deterministic_dropped}")
    print(f"Constant or empty baseline features dropped: {len(constant_dropped)}")
    if dry_run:
        _write_readme(output / "README.txt", train=train, test=test, metrics=None, best_params=None)
        print("Dry run requested; model fitting was skipped.")
        return {"train": train, "test": test, "cohort_audit": cohort_audit}

    print("Fitting train-only payment and tabular preprocessing...")
    preprocessor = make_rsf_preprocessor(train.X, seed=seed)
    X_train_transformed = preprocessor.fit_transform(train.X, train.y)
    X_test_transformed = preprocessor.transform(test.X)

    parameter_path = output / BEST_PARAMETERS_FILENAME
    if tune_hyperparameters:
        parameter_grid = build_rsf_tuning_grid(
            max_candidates=tuning_max_candidates,
            seed=seed,
        )
        print(
            f"Factory-transfer tuning {len(parameter_grid)} RSF candidates "
            f"with {tuning_trees} trees per directional fit...",
            flush=True,
        )
        tuning_results, best_params = tune_rsf_factory_transfer(
            train.X,
            train.y,
            parameter_grid=parameter_grid,
            n_estimators=tuning_trees,
            seed=seed,
            n_jobs=n_jobs,
        )
        tuning_results.to_csv(output / "hyperparameter_tuning.csv", index=False)
        parameter_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selection_method": "bidirectional_factory_transfer",
            "primary_metric": "Mean_Transfer_IBS",
            "tie_breaker": "Mean_Transfer_Uno_C",
            "candidate_count": len(parameter_grid),
            "tuning_trees": tuning_trees,
            "selected_parameters": best_params,

        }
        parameter_path.write_text(
            json.dumps(_json_ready(parameter_payload), indent=2), encoding="utf-8"
        )
        parameter_source = str(parameter_path)
        print(f"Selected and saved RSF parameters: {best_params}", flush=True)
    else:
        best_params, parameter_source = load_rsf_parameters(parameter_path)
        tuning_results = pd.DataFrame()
        print(f"Using RSF parameters from {parameter_source}: {best_params}", flush=True)

    tree_counts = _parse_tree_counts(stability_tree_counts, final_trees)
    print(f"Growing selected RSF through tree counts: {tree_counts}")
    forest, stability = fit_final_rsf_with_stability(
        X_train_transformed,
        train.y,
        params=best_params,
        tree_counts=tree_counts,
        seed=seed,
        n_jobs=n_jobs,
    )
    stability.to_csv(output / "oob_stability.csv", index=False)
    _save_line_plot(
        stability,
        "Trees",
        ["OOB_Harrell_C", "OOB_Uno_C"],
        output / "oob_stability.png",
        "OOB concordance",
    )


    print("Evaluating RSF discrimination, calibration, and uncertainty...")
    fitted_pipeline = Pipeline([("preprocessor", preprocessor), ("rsf", forest)])
    metrics, brier_curve, risk, survival_reported = evaluate_external_test(
        forest,
        X_test_transformed,
        train.y,
        test.y,
    )
    metrics.update(
        write_extended_performance_outputs(
            forest,
            X_test_transformed,
            train.y,
            test.y,
            output=output,
            slug="rsf",
            label="RSF",
            bootstrap_repeats=bootstrap_repeats,
            calibration_bins=calibration_bins,
            seed=seed,
        )
    )

    metrics.update(
        {
            "Final_OOB_Harrell_C": float(stability.iloc[-1]["OOB_Harrell_C"]),
            "Final_OOB_Uno_C": float(stability.iloc[-1]["OOB_Uno_C"]),
            "Final_Trees": int(stability.iloc[-1]["Trees"]),
        }
    )
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "rsf_metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    (output / "rsf_metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    brier_curve.to_csv(output / "brier_curve.csv", index=False)
    brier_curve.to_csv(output / "rsf_brier_curve.csv", index=False)
    _save_line_plot(
        brier_curve,
        "Time_Months",
        ["RSF_Brier", "Kaplan_Meier_Brier"],
        output / "brier_curve.png",
        "Brier score (lower is better)",
    )

    _, reported_times, _ = choose_evaluation_times(train.y, test.y)
    predictions = test.metadata.copy()
    predictions["rsf_risk_score"] = risk
    for column_number, time in enumerate(reported_times):
        predictions[f"survival_probability_{time:g}_months"] = survival_reported[:, column_number]
        predictions[f"turnover_probability_{time:g}_months"] = 1.0 - survival_reported[:, column_number]
    predictions.to_csv(output / "file3_survival_predictions.csv", index=False)

    artifacts.mkdir(parents=True, exist_ok=True)
    model_rows = [{"Model": "RSF", **metrics}]
    baseline_artifacts: dict[str, object] = {}

    if not skip_baselines:
        print("Scaling transformed covariates for the pooled discrete-time model...")
        baseline_scaler = StandardScaler(with_mean=False)
        X_train_baseline = baseline_scaler.fit_transform(X_train_transformed)
        X_test_baseline = baseline_scaler.transform(X_test_transformed)
        baseline_specs = [

            (
                "Pooled Discrete-Time",
                "pooled_discrete_time",
                PooledLogisticSurvival(
                    interval_months=discrete_interval_months,
                    C=discrete_C,
                    max_iter=discrete_max_iterations,
                    random_state=seed,
                    verbose=1,
                ),
                {
                    "interval_months": discrete_interval_months,
                    "C": discrete_C,
                    "max_iter": discrete_max_iterations,
                },
            ),
        ]
        for label, slug, model, parameters in baseline_specs:
            print(f"Fitting {label} model...")
            with fitting_progress(label, update_seconds=fit_progress_seconds):
                model.fit(X_train_baseline, train.y)
            print(f"Evaluating {label} discrimination, calibration, and uncertainty...")
            model_metrics, model_brier, _, _ = evaluate_external_test(
                model,
                X_test_baseline,
                train.y,
                test.y,
            )
            model_metrics.update(
                write_extended_performance_outputs(
                    model,
                    X_test_baseline,
                    train.y,
                    test.y,
                    output=output,
                    slug=slug,
                    label=label,
                    bootstrap_repeats=bootstrap_repeats,
                    calibration_bins=calibration_bins,
                    seed=seed,
                )
            )
            pd.DataFrame([model_metrics]).to_csv(output / f"{slug}_metrics.csv", index=False)
            (output / f"{slug}_metrics.json").write_text(
                json.dumps(_json_ready(model_metrics), indent=2), encoding="utf-8"
            )
            model_brier = model_brier.rename(columns={"RSF_Brier": f"{label}_Brier"})
            model_brier.to_csv(output / f"{slug}_brier_curve.csv", index=False)
            _save_line_plot(
                model_brier,
                "Time_Months",
                [f"{label}_Brier", "Kaplan_Meier_Brier"],
                output / f"{slug}_brier_curve.png",
                "Brier score (lower is better)",
            )
            model_rows.append({"Model": label, **model_metrics})
            baseline_artifacts[slug] = {
                "artifact_version": 1,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_type": label,
                "pipeline": Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("scale", baseline_scaler),
                        ("survival_model", model),
                    ]
                ),
                "feature_columns": list(train.X.columns),
                "parameters": parameters,
                "metrics": model_metrics,
                "cohort_audit": cohort_audit,
            }
            joblib.dump(baseline_artifacts[slug], artifacts / f"{slug}_survival.pkl")

    comparison = pd.DataFrame(model_rows)
    comparison.to_csv(output / "survival_model_comparison.csv", index=False)

    rsf_artifact = {
        "artifact_version": 4,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "RandomSurvivalForest",
        "pipeline": fitted_pipeline,
        "feature_columns": list(train.X.columns),
        "selected_parameters": best_params,
        "parameter_source": parameter_source,


        "censor_follow_up_months": censor_follow_up_months,
        "train_sources": ["file1", "file2"],
        "external_test_source": "file3",
        "metrics": metrics,
        "cohort_audit": cohort_audit,
        "reported_horizons_months": reported_times.tolist(),
    }
    artifact_path = artifacts / "random_survival_forest.pkl"
    joblib.dump(rsf_artifact, artifact_path)

    vimp_status = {
        "requested": not skip_permutation,
        "status": "skipped" if skip_permutation else "in_progress",
        "repeats": permutation_repeats,
        "note": "Existing VIMP files may belong to an earlier run when status is skipped.",
    }
    (output / "vimp_status.json").write_text(
        json.dumps(_json_ready(vimp_status), indent=2), encoding="utf-8"
    )
    if skip_permutation:
        print("Skipping RSF permutation VIMP as requested.")
    else:
        print(f"Computing file3 RSF permutation VIMP with {permutation_repeats} repeats...")
        c_vimp, ibs_vimp = permutation_vimp(
            fitted_pipeline,
            test.X,
            train.y,
            test.y,
            repeats=permutation_repeats,
            seed=seed,
            n_jobs=permutation_jobs,
        )
        c_vimp.to_csv(output / "permutation_vimp_ipcw_cindex.csv", index=False)
        ibs_vimp.to_csv(output / "permutation_vimp_ibs.csv", index=False)
        _save_vimp_plot(
            c_vimp,
            "C_Index_Decrease_Mean",
            "C_Index_Decrease_Std",
            output / "permutation_vimp_ipcw_cindex.png",
        )
        _save_vimp_plot(
            ibs_vimp,
            "IBS_Increase_Mean",
            "IBS_Increase_Std",
            output / "permutation_vimp_ibs.png",
        )
        vimp_status["status"] = "completed"
        (output / "vimp_status.json").write_text(
            json.dumps(_json_ready(vimp_status), indent=2), encoding="utf-8"
        )

    _write_readme(
        output / "README.txt",
        train=train,
        test=test,
        metrics=metrics,
        best_params=best_params,
    )
    print(f"Reports: {output}")
    print(f"RSF artifact: {artifact_path}")
    return {
        "metrics": metrics,
        "model_comparison": comparison,
        "best_params": best_params,
        "tuning_results": tuning_results,
        "stability": stability,
        "artifact_path": artifact_path,
        "parameter_source": parameter_source,

    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit and compare employee-level RSF and pooled discrete-time models."
    )
    parser.add_argument("--output-dir", default="output/rsf")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--censor-follow-up-months", type=int, default=DEFAULT_FOLLOW_UP_MONTHS)
    parser.add_argument("--tuning-trees", type=int, default=250)
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Run optional factory1<->factory2 RSF tuning and save the winner.",
    )
    parser.add_argument(
        "--tuning-max-candidates",
        type=int,
        default=24,
        help="Reproducible candidates sampled from the 144-combination grid; 0 runs all.",
    )
    parser.add_argument("--final-trees", type=int, default=750)
    parser.add_argument("--stability-tree-counts", default="100,250,500")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument("--discrete-interval-months", type=float, default=6.0)
    parser.add_argument("--discrete-c", type=float, default=1.0)
    parser.add_argument("--discrete-max-iterations", type=int, default=2000)
    parser.add_argument(
        "--fit-progress-seconds",
        type=float,
        default=5.0,
        help="Seconds between elapsed-time activity-bar updates during pooled fitting; 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--permutation-jobs", type=int, default=1)

    parser.add_argument(
        "--skip-vimp",
        "--skip-permutation",
        dest="skip_permutation",
        action="store_true",
        help="Refit and evaluate the RSF without running expensive permutation VIMP.",
    )
    parser.add_argument(
        "--skip-discrete-time",
        "--skip-baselines",
        dest="skip_baselines",
        action="store_true",
        help="Skip pooled discrete-time model fitting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_rsf_experiment(
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        censor_follow_up_months=args.censor_follow_up_months,
        tuning_trees=args.tuning_trees,
        final_trees=args.final_trees,
        stability_tree_counts=args.stability_tree_counts,
        permutation_repeats=args.permutation_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        calibration_bins=args.calibration_bins,



        discrete_interval_months=args.discrete_interval_months,
        discrete_C=args.discrete_c,
        discrete_max_iterations=args.discrete_max_iterations,
        fit_progress_seconds=args.fit_progress_seconds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        permutation_jobs=args.permutation_jobs,
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_max_candidates=args.tuning_max_candidates,
        skip_permutation=args.skip_permutation,
        skip_baselines=args.skip_baselines,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
