"""Additional models and censoring-aware evaluation for survival experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sksurv.functions import StepFunction
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


DETERMINISTIC_REDUNDANT_COLUMNS = {
    "tenure_years",       # vetek_months / 12
    "career_start_age",   # age - vetek_months / 12
    "is_new_employee",    # deterministic threshold of vetek_months
}


def _survival_target(events, times) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(events, dtype=bool),
        time=np.asarray(times, dtype=float),
        name_event="event",
        name_time="time",
    )


def evaluate_step_function(function: StepFunction, times: np.ndarray) -> np.ndarray:
    """Evaluate a survival step function, returning one before its first knot."""
    values = np.ones(len(times), dtype=float)
    supported = times >= function.x[0]
    if supported.any():
        values[supported] = function(times[supported])
    return values


def survival_probability_matrix(estimator, X, times: np.ndarray) -> np.ndarray:
    functions = estimator.predict_survival_function(X)
    return np.vstack([evaluate_step_function(function, times) for function in functions])


def _km_event_probability(y: np.ndarray, horizon: float) -> float:
    if len(y) == 0:
        return float("nan")
    km_times, km_survival = kaplan_meier_estimator(y["event"], y["time"])
    position = int(np.searchsorted(km_times, horizon, side="right") - 1)
    survival = 1.0 if position < 0 else float(km_survival[position])
    return 1.0 - survival


def calibration_table(
    y_test: np.ndarray,
    survival_probabilities: np.ndarray,
    horizons: Sequence[float],
    *,
    n_bins: int = 5,
    bootstrap_repeats: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Build horizon-specific KM calibration tables with bootstrap intervals."""
    if n_bins < 2:
        raise ValueError("Calibration requires at least two bins.")
    if survival_probabilities.shape != (len(y_test), len(horizons)):
        raise ValueError("Calibration probabilities do not match test rows and horizons.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for column, horizon in enumerate(horizons):
        predicted_risk = 1.0 - survival_probabilities[:, column]
        if not np.isfinite(predicted_risk).all():
            invalid = int((~np.isfinite(predicted_risk)).sum())
            raise ValueError(
                f"Calibration received {invalid} non-finite predictions at {float(horizon):g} months."
            )
        if np.unique(predicted_risk).size < 2:
            bin_codes = np.zeros(len(predicted_risk), dtype=int)
        else:
            bin_codes = pd.qcut(
                pd.Series(predicted_risk),
                q=min(n_bins, len(predicted_risk)),
                labels=False,
                duplicates="drop",
            ).to_numpy()
            if pd.isna(bin_codes).all():
                bin_codes = np.zeros(len(predicted_risk), dtype=int)
        valid_bins = sorted(int(value) for value in np.unique(bin_codes[~pd.isna(bin_codes)]))
        for bin_code in valid_bins:
            indices = np.flatnonzero(bin_codes == bin_code)
            observed = _km_event_probability(y_test[indices], float(horizon))
            bootstrap_values = []
            if bootstrap_repeats > 0 and len(indices) > 1:
                for _ in range(bootstrap_repeats):
                    sampled = rng.choice(indices, size=len(indices), replace=True)
                    value = _km_event_probability(y_test[sampled], float(horizon))
                    if np.isfinite(value):
                        bootstrap_values.append(value)
            if bootstrap_values:
                lower, upper = np.percentile(bootstrap_values, [2.5, 97.5])
            else:
                lower = upper = float("nan")
            rows.append(
                {
                    "Horizon_Months": float(horizon),
                    "Risk_Bin": bin_code + 1,
                    "Employees": len(indices),
                    "Predicted_Risk_Mean": float(np.mean(predicted_risk[indices])),
                    "Observed_Risk_KM": observed,
                    "Observed_Risk_CI_Lower": float(lower),
                    "Observed_Risk_CI_Upper": float(upper),
                    "Bin_Calibration_Difference": observed - float(np.mean(predicted_risk[indices])),
                }
            )
    return pd.DataFrame(rows)


def save_calibration_plot(table: pd.DataFrame, path: str | Path, *, title: str) -> None:
    horizons = sorted(table["Horizon_Months"].unique())
    columns = min(2, len(horizons))
    rows = int(np.ceil(len(horizons) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for axis, horizon in zip(axes.flat, horizons):
        data = table.loc[table["Horizon_Months"].eq(horizon)].sort_values("Predicted_Risk_Mean")
        lower_error = data["Observed_Risk_KM"] - data["Observed_Risk_CI_Lower"]
        upper_error = data["Observed_Risk_CI_Upper"] - data["Observed_Risk_KM"]
        axis.errorbar(
            data["Predicted_Risk_Mean"],
            data["Observed_Risk_KM"],
            yerr=np.vstack([lower_error.clip(lower=0), upper_error.clip(lower=0)]),
            marker="o",
            capsize=3,
            label="KM observed",
        )
        maximum = max(
            0.05,
            float(data[["Predicted_Risk_Mean", "Observed_Risk_KM"]].max().max()) * 1.10,
        )
        axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", label="Ideal")
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
        axis.set_title(f"{horizon:g}-month calibration")
        axis.set_xlabel("Mean predicted turnover risk")
        axis.set_ylabel("Observed turnover risk")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(horizons):]:
        axis.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def time_dependent_auc_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_probabilities: np.ndarray,
    times: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    event_probabilities = 1.0 - survival_probabilities
    auc_values, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        event_probabilities,
        np.asarray(times, dtype=float),
    )
    metrics = {"Mean_Time_Dependent_AUC": float(mean_auc)}
    for time, value in zip(times, auc_values):
        metrics[f"Dynamic_AUC_{float(time):g}_Months"] = float(value)
    table = pd.DataFrame({"Time_Months": times, "Dynamic_AUC": auc_values})
    return metrics, table


def bootstrap_performance_intervals(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    survival_grid: np.ndarray,
    grid_times: np.ndarray,
    survival_horizons: np.ndarray,
    horizons: np.ndarray,
    *,
    tau: float,
    repeats: int = 500,
    seed: int = 42,
    progress_label: str = "Model",
) -> pd.DataFrame:
    """External employee bootstrap CIs without refitting the model."""
    if repeats < 1:
        return pd.DataFrame(columns=["Metric", "Estimate", "CI_Lower", "CI_Upper", "Valid_Bootstraps"])

    def calculate(indices: np.ndarray) -> dict[str, float]:
        y = y_test[indices]
        risk = risk_scores[indices]
        grid_probabilities = survival_grid[indices]
        horizon_probabilities = survival_horizons[indices]
        values = {
            "Harrell_C": float(concordance_index_censored(y["event"], y["time"], risk)[0]),
            "Uno_IPCW_C": float(concordance_index_ipcw(y_train, y, risk, tau=tau)[0]),
            "IBS": float(integrated_brier_score(y_train, y, grid_probabilities, grid_times)),
        }
        auc_values, mean_auc = cumulative_dynamic_auc(
            y_train,
            y,
            1.0 - horizon_probabilities,
            horizons,
        )
        values["Mean_Time_Dependent_AUC"] = float(mean_auc)
        _, horizon_brier = brier_score(y_train, y, horizon_probabilities, horizons)
        for time, auc in zip(horizons, auc_values):
            values[f"Dynamic_AUC_{float(time):g}_Months"] = float(auc)
        for time, score in zip(horizons, horizon_brier):
            values[f"Brier_{float(time):g}_Months"] = float(score)
        return values

    full_indices = np.arange(len(y_test))
    estimates = calculate(full_indices)
    samples: dict[str, list[float]] = {name: [] for name in estimates}
    rng = np.random.default_rng(seed)
    update_every = max(repeats // 10, 1)
    for repeat in range(1, repeats + 1):
        indices = rng.integers(0, len(y_test), size=len(y_test))
        try:
            values = calculate(indices)
        except ValueError:
            continue
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(value)
        if repeat % update_every == 0 or repeat == repeats:
            print(f"[{progress_label} bootstrap] {repeat}/{repeats} ({repeat / repeats:.0%})", flush=True)

    rows = []
    for name, estimate in estimates.items():
        valid = np.asarray(samples[name], dtype=float)
        if len(valid):
            lower, upper = np.percentile(valid, [2.5, 97.5])
        else:
            lower = upper = float("nan")
        rows.append(
            {
                "Metric": name,
                "Estimate": estimate,
                "CI_Lower": float(lower),
                "CI_Upper": float(upper),
                "Valid_Bootstraps": len(valid),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class PooledLogisticSurvival(BaseEstimator):
    """Pooled discrete-time logistic-hazard survival model."""

    interval_months: float = 6.0
    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 42
    verbose: int = 0

    def _validate_interval(self) -> None:
        if self.interval_months <= 0:
            raise ValueError("interval_months must be positive.")

    def _interval_design(self, X, interval_indices: np.ndarray):
        X_sparse = sparse.csr_matrix(X, dtype=np.float32)
        interval_rows = np.arange(len(interval_indices))
        interval_data = np.ones(len(interval_indices), dtype=np.float32)
        indicators = sparse.csr_matrix(
            (interval_data, (interval_rows, interval_indices)),
            shape=(len(interval_indices), self.n_intervals_),
        )
        return sparse.hstack([X_sparse, indicators], format="csr")

    def fit(self, X, y):
        self._validate_interval()
        times = np.asarray(y["time"], dtype=float)
        events = np.asarray(y["event"], dtype=bool)
        maximum = float(np.max(times))
        n_intervals = max(int(np.ceil(maximum / self.interval_months)), 1)
        self.interval_edges_ = np.arange(n_intervals + 1, dtype=float) * self.interval_months
        self.n_intervals_ = n_intervals

        if self.verbose:
            print(
                f"[Pooled Discrete-Time fit] stage 1/3: expanding {len(times):,} employees "
                f"into {n_intervals} risk intervals.",
                flush=True,
            )
        repeated_subjects = []
        interval_indices = []
        interval_events = []
        for subject, (time, event) in enumerate(zip(times, events)):
            last_interval = min(
                int(np.searchsorted(self.interval_edges_[1:], time, side="left")),
                n_intervals - 1,
            )
            subject_intervals = np.arange(last_interval + 1, dtype=int)
            repeated_subjects.extend([subject] * len(subject_intervals))
            interval_indices.extend(subject_intervals.tolist())
            outcomes = np.zeros(len(subject_intervals), dtype=np.int8)
            if event:
                outcomes[-1] = 1
            interval_events.extend(outcomes.tolist())

        repeated_subjects = np.asarray(repeated_subjects, dtype=int)
        interval_indices = np.asarray(interval_indices, dtype=int)
        if self.verbose:
            print(
                f"[Pooled Discrete-Time fit] stage 2/3: building "
                f"{len(repeated_subjects):,} person-period rows.",
                flush=True,
            )
        X_repeated = X[repeated_subjects] if sparse.issparse(X) else np.asarray(X)[repeated_subjects]
        design = self._interval_design(X_repeated, interval_indices)
        if self.verbose:
            print(
                f"[Pooled Discrete-Time fit] stage 3/3: optimizing logistic hazard model "
                f"(maximum {self.max_iter:,} iterations).",
                flush=True,
            )
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(design, np.asarray(interval_events, dtype=np.int8))
        if self.verbose:
            iterations = int(np.max(self.model_.n_iter_))
            print(
                f"[Pooled Discrete-Time fit] optimizer finished after {iterations:,} iterations.",
                flush=True,
            )
        self.n_features_in_ = np.shape(X)[1]
        return self

    def _hazards(self, X) -> np.ndarray:
        n_samples = np.shape(X)[0]
        repeated = np.repeat(np.arange(n_samples), self.n_intervals_)
        intervals = np.tile(np.arange(self.n_intervals_), n_samples)
        X_repeated = X[repeated] if sparse.issparse(X) else np.asarray(X)[repeated]
        design = self._interval_design(X_repeated, intervals)
        hazards = self.model_.predict_proba(design)[:, 1]
        return np.clip(hazards.reshape(n_samples, self.n_intervals_), 1e-7, 1 - 1e-7)

    def predict(self, X) -> np.ndarray:
        hazards = self._hazards(X)
        return np.sum(-np.log1p(-hazards), axis=1)

    def predict_survival_function(self, X):
        survival = np.cumprod(1.0 - self._hazards(X), axis=1)
        x = np.concatenate([[0.0], self.interval_edges_[1:]])
        return [
            StepFunction(x, np.concatenate([[1.0], values]), domain=(0.0, float(x[-1])))
            for values in survival
        ]
