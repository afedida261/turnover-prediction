"""Train the EDA-driven final turnover models.

Training sources: file1 + file2.
External test source: file3 (never used for fitting, threshold selection, or
model selection).

Class imbalance is handled only through native estimator parameters:
``class_weight`` for Logistic Regression/Random Forest and
``scale_pos_weight`` for XGBoost. No resampling, IPW, focal loss, or synthetic
examples are used.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.imputations import PAYMENT_COLUMNS, SimilarEmployeePaymentImputer
from src.preprocess import PreparedTurnoverData, prepare_turnover_data, signed_log1p


SEED = 42
PAYMENT_KEYWORDS = ("payment", "salary", "sahar")
LOGISTIC_REDUNDANT_PREFIXES = (
    "Median_Payment",
    "Median_omes",
    "stdevp_illness",
)
MODEL_CONTEXT_COLUMNS = {"source", "year_date"}


@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    payment_strategy: str

    @property
    def name(self) -> str:
        return f"{self.model_name}__{self.payment_strategy}"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - old sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def is_payment_feature(column: str) -> bool:
    text = str(column).lower()
    return any(keyword in text for keyword in PAYMENT_KEYWORDS)


def candidate_feature_columns(X: pd.DataFrame, spec: CandidateSpec) -> list[str]:
    columns = list(X.columns)
    if spec.payment_strategy == "no_payment":
        columns = [column for column in columns if not is_payment_feature(column)]
    if spec.model_name == "Logistic Regression":
        columns = [
            column for column in columns
            if not any(str(column).startswith(prefix) for prefix in LOGISTIC_REDUNDANT_PREFIXES)
        ]
    return columns


def _model_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    usable = [
        column for column in frame.columns
        if column not in MODEL_CONTEXT_COLUMNS
        and not pd.api.types.is_datetime64_any_dtype(frame[column])
    ]
    numeric = [
        column for column in usable
        if pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical = [column for column in usable if column not in numeric]
    return numeric, categorical


def make_tabular_transformer(
    X: pd.DataFrame,
    *,
    model_name: str,
    native_numeric_missing: bool,
    payment_indicators: bool,
) -> ColumnTransformer:
    numeric, categorical = _model_columns(X)
    if payment_indicators:
        numeric = list(dict.fromkeys(
            numeric + [f"{column}_was_missing" for column in PAYMENT_COLUMNS if column in X.columns]
        ))

    if model_name == "Logistic Regression":
        numeric_transformer = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            (
                "signed_log",
                FunctionTransformer(
                    signed_log1p,
                    feature_names_out="one-to-one",
                    validate=False,
                ),
            ),
            ("scale", StandardScaler(with_mean=False)),
        ])
    elif native_numeric_missing:
        numeric_transformer = "passthrough"
    else:
        numeric_transformer = SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    categorical_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric),
            ("categorical", categorical_transformer, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def make_estimator(model_name: str, y_train: pd.Series, seed: int):
    positives = max(int(pd.Series(y_train).eq(1).sum()), 1)
    negatives = max(int(pd.Series(y_train).eq(0).sum()), 1)
    scale_pos_weight = negatives / positives

    if model_name == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=3,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def make_candidate_pipeline(spec: CandidateSpec, X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    use_learned_imputation = spec.payment_strategy == "learned_imputation"
    native_missing = spec.payment_strategy == "native_missing"
    steps = []
    if use_learned_imputation:
        steps.append(("payment_imputer", SimilarEmployeePaymentImputer(random_state=seed)))
    steps.extend([
        (
            "tabular",
            make_tabular_transformer(
                X,
                model_name=spec.model_name,
                native_numeric_missing=native_missing,
                payment_indicators=use_learned_imputation,
            ),
        ),
        ("model", make_estimator(spec.model_name, y, seed)),
    ])
    return Pipeline(steps)


def source_stratified_group_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out employees separately within file1 and file2."""
    group_frame = pd.DataFrame({
        "group": groups.astype(str).to_numpy(),
        "target": y.astype(int).to_numpy(),
        "source": X["source"].astype(str).to_numpy(),
    })
    group_labels = (
        group_frame.groupby("group", sort=False)
        .agg(target=("target", "max"), source=("source", "first"))
    )

    train_groups = []
    validation_groups = []
    for source, source_groups in group_labels.groupby("source"):
        stratify = source_groups["target"] if source_groups["target"].value_counts().min() >= 2 else None
        source_train, source_validation = train_test_split(
            source_groups.index.to_numpy(),
            test_size=validation_size,
            random_state=seed,
            stratify=stratify,
        )
        train_groups.extend(source_train)
        validation_groups.extend(source_validation)

    train_mask = groups.astype(str).isin(train_groups).to_numpy()
    validation_mask = groups.astype(str).isin(validation_groups).to_numpy()
    if np.any(train_mask & validation_mask):
        raise AssertionError("Employee leaked across train and validation.")
    return train_mask, validation_mask


def positive_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(X)
    return probabilities[:, 1] if probabilities.ndim == 2 else probabilities


def best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    return float(thresholds[best[-1]]) if len(best) else 0.5


def top_k_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    ranked = pd.DataFrame({"target": np.asarray(y_true), "probability": probabilities})
    ranked = ranked.sort_values("probability", ascending=False)
    positives = float(ranked["target"].sum())
    metrics = {}
    for fraction in (0.10, 0.20, 0.30):
        count = max(1, int(np.ceil(len(ranked) * fraction)))
        captured = float(ranked.head(count)["target"].sum())
        label = int(fraction * 100)
        metrics[f"Recall@Top{label}%"] = captured / positives if positives else 0.0
        metrics[f"Precision@Top{label}%"] = captured / count
    return metrics


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= threshold
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    metrics = {
        "AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(y, probabilities),
        "F1": f1_score(y, predictions, zero_division=0),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y, predictions),
        "Log_Loss": log_loss(y, clipped),
        "Brier": brier_score_loss(y, probabilities),
    }
    metrics.update(top_k_metrics(y, probabilities))
    return metrics


def validation_result_row(
    spec: CandidateSpec,
    y_validation: pd.Series,
    probabilities: np.ndarray,
) -> tuple[dict[str, float | str], float]:
    threshold = best_f1_threshold(y_validation, probabilities)
    tuned = evaluate_probabilities(y_validation, probabilities, threshold)
    default = evaluate_probabilities(y_validation, probabilities, 0.5)
    row: dict[str, float | str] = {
        "Candidate": spec.name,
        "Model": spec.model_name,
        "Payment_Strategy": spec.payment_strategy,
        "Validation_Threshold": threshold,
    }
    row.update({f"Val_{key}": value for key, value in tuned.items()})
    row.update({
        "Val_F1@0.5": default["F1"],
        "Val_Precision@0.5": default["Precision"],
        "Val_Recall@0.5": default["Recall"],
    })
    return row, threshold


def test_result_columns(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    tuned = evaluate_probabilities(y_test, probabilities, threshold)
    default = evaluate_probabilities(y_test, probabilities, 0.5)
    result = {f"Test_{key}": value for key, value in tuned.items()}
    result.update({
        "Test_F1@0.5": default["F1"],
        "Test_Precision@0.5": default["Precision"],
        "Test_Recall@0.5": default["Recall"],
    })
    return result


def feature_importance_frame(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    try:
        names = pipeline.named_steps["tabular"].get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance", "Direction"])
    model = pipeline.named_steps["model"]
    if model_name == "Logistic Regression":
        values = model.coef_[0]
        return (
            pd.DataFrame({
                "Feature": names,
                "Importance": np.abs(values),
                "Direction": np.sign(values),
                "Coefficient": values,
            })
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    values = getattr(model, "feature_importances_", None)
    if values is None or len(values) != len(names):
        return pd.DataFrame(columns=["Feature", "Importance", "Direction"])
    return (
        pd.DataFrame({"Feature": names, "Importance": values, "Direction": np.nan})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def candidate_specs() -> list[CandidateSpec]:
    specs = []
    for model_name in ("Logistic Regression", "Random Forest", "XGBoost"):
        specs.append(CandidateSpec(model_name, "learned_imputation"))
        specs.append(CandidateSpec(model_name, "no_payment"))
    specs.append(CandidateSpec("XGBoost", "native_missing"))
    return specs


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    prepared: PreparedTurnoverData,
    best_candidate: str,
) -> None:
    columns = [
        "Candidate",
        "Validation_Threshold",
        "Val_AUC",
        "Val_PR_AUC",
        "Val_F1",
        "Val_Precision",
        "Val_Recall",
        "Val_Recall@Top20%",
        "Val_Precision@Top20%",
        "Test_AUC",
        "Test_PR_AUC",
        "Test_F1",
        "Test_Precision",
        "Test_Recall",
        "Test_Recall@Top20%",
        "Test_Precision@Top20%",
        "Test_Log_Loss",
        "Test_Brier",
    ]
    display = comparison[[column for column in columns if column in comparison]].copy()
    lines = [
        "FINAL EDA-DRIVEN TURNOVER MODELS",
        "=" * 88,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Best candidate selected on validation PR-AUC: {best_candidate}",
        "",
        "Data",
        f"- file1 + file2 employee-period rows: {len(prepared.X_train):,}",
        f"- file3 external-test rows: {len(prepared.X_test):,}",
        f"- invalid rows removed: {prepared.cleaning_audit.removed_rows:,}",
        f"- age-above-75 rows retained: {prepared.cleaning_audit.retained_age_above_75_rows:,}",
        "",
        "Leakage and imbalance controls",
        "- Employee-grouped validation, stratified separately within file1 and file2.",
        "- File3 used only after validation model/threshold selection.",
        "- Native class weighting only: class_weight or scale_pos_weight.",
        "- No resampling, SMOTE, IPW, focal loss, or synthetic examples.",
        "- Validation threshold maximizes F1; top-k metrics do not depend on that threshold.",
        "",
        "Payment strategies",
        "- learned_imputation: similar-employee payment imputer fitted within the train fold.",
        "- native_missing: XGBoost receives numeric NaNs directly.",
        "- no_payment: all payment/salary features and their histories are removed.",
        "",
        "Model comparison",
        display.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_final_models(
    *,
    output_dir: str | Path = "output/final",
    artifact_dir: str | Path = "artifacts",
    seed: int = SEED,
) -> pd.DataFrame:
    output = Path(output_dir)
    artifacts = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    prepared = prepare_turnover_data()
    train_mask, validation_mask = source_stratified_group_validation_split(
        prepared.X_train,
        prepared.y_train,
        prepared.train_groups,
        validation_size=0.20,
        seed=seed,
    )
    X_inner = prepared.X_train.loc[train_mask]
    y_inner = prepared.y_train.loc[train_mask]
    X_validation = prepared.X_train.loc[validation_mask]
    y_validation = prepared.y_train.loc[validation_mask]

    validation_rows = []
    thresholds = {}
    specs = candidate_specs()
    print(
        f"Inner train: {len(X_inner):,} rows; validation: {len(X_validation):,} rows; "
        f"external test: {len(prepared.X_test):,} rows"
    )

    for spec in specs:
        print(f"Validating {spec.name}...")
        columns = candidate_feature_columns(X_inner, spec)
        pipeline = make_candidate_pipeline(spec, X_inner[columns], y_inner, seed)
        pipeline.fit(X_inner[columns], y_inner)
        probabilities = positive_probabilities(pipeline, X_validation[columns])
        row, threshold = validation_result_row(spec, y_validation, probabilities)
        row["Feature_Count_Before_Encoding"] = len(columns)
        validation_rows.append(row)
        thresholds[spec.name] = threshold

    validation_comparison = pd.DataFrame(validation_rows).sort_values(
        ["Val_PR_AUC", "Val_AUC"], ascending=False
    ).reset_index(drop=True)
    best_candidate = str(validation_comparison.iloc[0]["Candidate"])
    print(f"Selected on validation PR-AUC: {best_candidate}")

    final_rows = []
    artifact_paths = {}
    best_predictions = None
    for spec in specs:
        print(f"Refitting {spec.name} on all file1+file2 rows...")
        columns = candidate_feature_columns(prepared.X_train, spec)
        pipeline = make_candidate_pipeline(spec, prepared.X_train[columns], prepared.y_train, seed)
        pipeline.fit(prepared.X_train[columns], prepared.y_train)
        test_probabilities = positive_probabilities(pipeline, prepared.X_test[columns])
        threshold = thresholds[spec.name]

        validation_row = validation_comparison.loc[
            validation_comparison["Candidate"].eq(spec.name)
        ].iloc[0].to_dict()
        validation_row.update(test_result_columns(prepared.y_test, test_probabilities, threshold))
        final_rows.append(validation_row)

        importance = feature_importance_frame(pipeline, spec.model_name)
        importance.to_csv(output / f"feature_importance_{slugify(spec.name)}.csv", index=False)

        artifact = {
            "artifact_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "candidate": spec.name,
            "model_name": spec.model_name,
            "payment_strategy": spec.payment_strategy,
            "pipeline": pipeline,
            "feature_columns": columns,
            "decision_threshold": threshold,
            "target_column": "leave_ind",
            "train_sources": ["file1", "file2"],
            "external_test_source": "file3",
            "native_class_balance": (
                "class_weight=balanced" if spec.model_name == "Logistic Regression"
                else "class_weight=balanced_subsample" if spec.model_name == "Random Forest"
                else "scale_pos_weight=negative/positive"
            ),
            "cleaning_audit": asdict(prepared.cleaning_audit),
            "validation_metrics": {
                key: value for key, value in validation_row.items() if str(key).startswith("Val_")
            },
            "test_metrics": {
                key: value for key, value in validation_row.items() if str(key).startswith("Test_")
            },
        }
        artifact_path = artifacts / f"final_{slugify(spec.name)}.pkl"
        joblib.dump(artifact, artifact_path)
        artifact_paths[spec.name] = artifact_path

        if spec.name == best_candidate:
            best_predictions = test_probabilities

    comparison = pd.DataFrame(final_rows).sort_values(
        ["Val_PR_AUC", "Val_AUC"], ascending=False
    ).reset_index(drop=True)
    comparison.insert(0, "Selected_Best", comparison["Candidate"].eq(best_candidate))
    comparison.to_csv(output / "model_comparison.csv", index=False)

    best_path = artifacts / "final_best_model.pkl"
    shutil.copy2(artifact_paths[best_candidate], best_path)

    predictions = prepared.test_frame[
        [
            column for column in [
                "source_employee_id",
                "fictive_employee",
                "calc_month",
                "leave_ind",
                "age",
                "vetek_months",
                "contract_type",
            ]
            if column in prepared.test_frame.columns
        ]
    ].copy()
    predictions["turnover_probability"] = best_predictions
    best_threshold = thresholds[best_candidate]
    predictions["turnover_prediction"] = (
        predictions["turnover_probability"] >= best_threshold
    ).astype(int)
    predictions["risk_rank"] = predictions["turnover_probability"].rank(
        method="first", ascending=False
    ).astype(int)
    predictions.to_excel(output / "file3_predictions.xlsx", index=False)

    split_summary = pd.DataFrame([
        {
            "Split": "inner_train",
            "Rows": len(y_inner),
            "Employees": prepared.train_groups.loc[train_mask].nunique(),
            "Positive_Rows": int(y_inner.sum()),
            "Positive_Rate": float(y_inner.mean()),
        },
        {
            "Split": "validation",
            "Rows": len(y_validation),
            "Employees": prepared.train_groups.loc[validation_mask].nunique(),
            "Positive_Rows": int(y_validation.sum()),
            "Positive_Rate": float(y_validation.mean()),
        },
        {
            "Split": "file3_external_test",
            "Rows": len(prepared.y_test),
            "Employees": prepared.test_groups.nunique(),
            "Positive_Rows": int(prepared.y_test.sum()),
            "Positive_Rate": float(prepared.y_test.mean()),
        },
    ])
    split_summary.to_csv(output / "split_summary.csv", index=False)
    (output / "cleaning_audit.json").write_text(
        json.dumps(asdict(prepared.cleaning_audit), indent=2), encoding="utf-8"
    )
    write_report(output / "README.txt", comparison, prepared, best_candidate)

    print(comparison[
        [
            "Selected_Best",
            "Candidate",
            "Val_PR_AUC",
            "Val_AUC",
            "Test_PR_AUC",
            "Test_AUC",
            "Test_Recall@Top20%",
            "Test_Precision@Top20%",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Reports: {output}")
    print(f"Artifacts: {artifacts}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final EDA-driven turnover models.")
    parser.add_argument("--output-dir", default="output/final")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_final_models(
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

