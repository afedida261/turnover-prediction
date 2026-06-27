"""Leakage-safe, similar-employee payment imputation.

The imputer is fitted on observed payment values from file1 and file2 only.
File3 is deliberately transform-only while it remains the external test set.
When doing internal cross-validation, place this transformer inside the model
pipeline so it is refitted on each training fold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.datasets import read_excel_with_header_detection


PAYMENT_COLUMNS = (
    "avg_Payment",
    "Median_Payment",
    "stdevp_Payment",
    "salary_change",
)

# Deliberately excludes outcome metadata, IDs, very high-cardinality codes, and
# payment fields. These are characteristics known at the prediction period.
DEFAULT_PAYMENT_PREDICTORS = (
    "source",
    "year_date",
    "contract_type",
    "vetek_months",
    "age",
    "Seif",
    "Maamad",
    "Maneger",
    "Mahala",
    "WorkHours",
    "Sahar",
    "MZV_Flag",
    "hedrut",
    "children",
    "avg_illness",
    "stdevp_illness",
    "Median_illness",
    "avg_omes",
    "stdevp_omes",
    "Median_omes",
    "count_managers",
)

NONNEGATIVE_PAYMENT_COLUMNS = {
    "avg_Payment",
    "Median_Payment",
    "stdevp_Payment",
}


def _canonicalize_known_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept both raw and canonical EDA column names."""
    aliases = {
        "TeurGroupHscm": "contract_type",
        "EMP_Matzav_Mishpachti": "marital_status",
        "שינוי השכר בשקלים": "salary_change",
    }
    return frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})


def _make_one_hot_encoder() -> OneHotEncoder:
    # The repository targets current sklearn, but this fallback keeps the
    # script usable on installations older than 1.2.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _target_to_model_scale(values: pd.Series, column: str) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if column in NONNEGATIVE_PAYMENT_COLUMNS:
        return np.log1p(np.clip(array, 0, None))
    return array


def _target_from_model_scale(values: np.ndarray, column: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if column in NONNEGATIVE_PAYMENT_COLUMNS:
        return np.clip(np.expm1(array), 0, None)
    return array


@dataclass(frozen=True)
class PaymentModelInfo:
    column: str
    observed_rows: int
    fallback_value: float
    predictor_columns: tuple[str, ...]


class SimilarEmployeePaymentImputer(BaseEstimator, TransformerMixin):
    """Impute payment fields using observed, similar file1/file2 employees.

    A separate HistGradientBoosting model is fitted for each payment column.
    Predictors are imputed/one-hot encoded inside each target-specific model.
    Original missingness is retained as ``<column>_was_missing``.
    """

    def __init__(
        self,
        payment_columns: Iterable[str] = PAYMENT_COLUMNS,
        predictor_columns: Iterable[str] = DEFAULT_PAYMENT_PREDICTORS,
        source_column: str = "source",
        forbidden_fit_sources: Iterable[str] = ("file3",),
        min_observed_rows: int = 50,
        random_state: int = 42,
        max_iter: int = 150,
    ):
        self.payment_columns = tuple(payment_columns)
        self.predictor_columns = tuple(predictor_columns)
        self.source_column = source_column
        self.forbidden_fit_sources = tuple(forbidden_fit_sources)
        self.min_observed_rows = min_observed_rows
        self.random_state = random_state
        self.max_iter = max_iter

    def _check_fit_sources(self, frame: pd.DataFrame) -> None:
        if self.source_column not in frame.columns:
            return
        present = set(frame[self.source_column].dropna().astype(str).unique())
        forbidden = present.intersection(self.forbidden_fit_sources)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(
                f"Payment imputer cannot be fitted on external test source(s): {names}. "
                "Fit on file1/file2 and call transform() on file3."
            )

    @staticmethod
    def _build_model(frame: pd.DataFrame, predictors: list[str], random_state: int, max_iter: int) -> Pipeline:
        numeric = [
            c for c in predictors
            if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
        ]
        categorical = [c for c in predictors if c not in numeric]

        transformers = []
        if numeric:
            transformers.append((
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ))
        if categorical:
            transformers.append((
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", _make_one_hot_encoder()),
                ]),
                categorical,
            ))

        preprocessing = ColumnTransformer(transformers, remainder="drop")
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=max_iter,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )
        return Pipeline([("features", preprocessing), ("regressor", regressor)])

    def fit(self, X: pd.DataFrame, y=None):
        frame = _canonicalize_known_columns(pd.DataFrame(X).copy())
        self._check_fit_sources(frame)

        self.models_: dict[str, Pipeline | None] = {}
        self.model_info_: dict[str, PaymentModelInfo] = {}
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)

        predictors = [c for c in self.predictor_columns if c in frame.columns]
        if not predictors:
            raise ValueError("None of the configured payment-imputation predictors are present.")

        for column in self.payment_columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            observed = values.notna() & np.isfinite(values)
            if column in NONNEGATIVE_PAYMENT_COLUMNS:
                observed &= values.ge(0)

            fallback = float(values.loc[observed].median()) if observed.any() else 0.0
            model = None
            if int(observed.sum()) >= self.min_observed_rows:
                model = self._build_model(frame, predictors, self.random_state, self.max_iter)
                model.fit(frame.loc[observed, predictors], _target_to_model_scale(values.loc[observed], column))

            self.models_[column] = model
            self.model_info_[column] = PaymentModelInfo(
                column=column,
                observed_rows=int(observed.sum()),
                fallback_value=fallback,
                predictor_columns=tuple(predictors),
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("Payment imputer must be fitted before transform().")

        frame = _canonicalize_known_columns(pd.DataFrame(X).copy())
        for column, info in self.model_info_.items():
            if column not in frame.columns:
                frame[column] = np.nan
            values = pd.to_numeric(frame[column], errors="coerce")
            missing = values.isna() | ~np.isfinite(values)
            frame[f"{column}_was_missing"] = missing.astype("int8")

            if missing.any():
                model = self.models_[column]
                if model is None:
                    predictions = np.full(int(missing.sum()), info.fallback_value)
                else:
                    predictors = list(info.predictor_columns)
                    predict_frame = frame.reindex(columns=predictors).loc[missing]
                    predictions = _target_from_model_scale(model.predict(predict_frame), column)
                values.loc[missing] = predictions
            frame[column] = values
        return frame

    def get_feature_names_out(self, input_features=None):
        base = list(input_features if input_features is not None else self.feature_names_in_)
        indicators = [f"{c}_was_missing" for c in self.model_info_ if f"{c}_was_missing" not in base]
        return np.asarray(base + indicators, dtype=object)


def cross_validate_payment_imputer(
    frame: pd.DataFrame,
    *,
    group_column: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure imputation error without mixing an employee across folds."""
    data = _canonicalize_known_columns(frame.copy()).reset_index(drop=True)
    if group_column not in data.columns:
        raise ValueError(f"Group column '{group_column}' is missing.")

    unique_groups = data[group_column].nunique()
    splits = min(n_splits, unique_groups)
    if splits < 2:
        raise ValueError("At least two employee groups are required for validation.")

    rows = []
    group_kfold = GroupKFold(n_splits=splits)
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(data, groups=data[group_column]), start=1):
        train = data.iloc[train_idx]
        validation = data.iloc[val_idx]
        imputer = SimilarEmployeePaymentImputer(random_state=random_state)
        imputer.fit(train)

        masked = validation.copy()
        observed_masks: dict[str, pd.Series] = {}
        actual_values: dict[str, pd.Series] = {}
        for column in PAYMENT_COLUMNS:
            if column not in masked.columns:
                continue
            # Copy because assigning NaN into ``masked`` can otherwise mutate
            # the Series view retained for scoring.
            actual = pd.to_numeric(masked[column], errors="coerce").copy()
            observed = actual.notna() & np.isfinite(actual)
            if column in NONNEGATIVE_PAYMENT_COLUMNS:
                observed &= actual.ge(0)
            observed_masks[column] = observed
            actual_values[column] = actual
            masked.loc[observed, column] = np.nan

        predicted = imputer.transform(masked)
        for column, observed in observed_masks.items():
            source_slices = {"overall": observed}
            if "source" in validation.columns:
                source_slices.update({
                    str(source): observed & validation["source"].eq(source)
                    for source in validation["source"].dropna().unique()
                })
            for source, score_mask in source_slices.items():
                if not score_mask.any():
                    continue
                actual = actual_values[column].loc[score_mask].to_numpy(dtype=float)
                estimate = predicted.loc[score_mask, column].to_numpy(dtype=float)
                rows.append({
                    "fold": fold,
                    "source": source,
                    "column": column,
                    "rows": len(actual),
                    "mae": mean_absolute_error(actual, estimate),
                    "rmse": mean_squared_error(actual, estimate) ** 0.5,
                    "median_absolute_error": float(np.median(np.abs(actual - estimate))),
                })
    return pd.DataFrame(rows)


def load_training_sources(paths: Mapping[str, str | Path] | None = None) -> pd.DataFrame:
    paths = paths or {"file1": "data/file1.xlsx", "file2": "data/file2.xlsx"}
    frames = []
    for source, path in paths.items():
        frame, _ = read_excel_with_header_detection(path)
        frame = _canonicalize_known_columns(frame)
        frame["source"] = source
        frame["source_employee_id"] = source + ":" + frame["fictive_employee"].astype(str)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    invalid = (
        pd.to_numeric(combined["avg_illness"], errors="coerce").lt(0)
        | pd.to_numeric(combined["avg_Payment"], errors="coerce").le(0)
        | pd.to_numeric(combined["avg_omes"], errors="coerce").le(0)
        | pd.to_numeric(combined["age"], errors="coerce").lt(18)
        | pd.to_numeric(combined["vetek_months"], errors="coerce").lt(0)
    )
    return combined.loc[~invalid.fillna(False)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate similar-employee payment imputation on file1/file2.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="output/preprocessed/payment_imputation_cv.csv")
    args = parser.parse_args()

    train = load_training_sources()
    report = cross_validate_payment_imputer(
        train,
        group_column="source_employee_id",
        n_splits=args.folds,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(
        report.groupby(["source", "column"])[["mae", "rmse", "median_absolute_error"]]
        .mean()
        .round(2)
    )
    print(f"Saved fold-level report to {output}")


if __name__ == "__main__":
    main()
