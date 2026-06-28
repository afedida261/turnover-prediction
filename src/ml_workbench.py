"""ML workbench backend for flexible final-workflow experiments."""

from __future__ import annotations

import contextlib
import io
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.final_modeling import (
    CandidateSpec,
    best_f1_threshold,
    candidate_feature_columns,
    candidate_specs,
    evaluate_probabilities,
    feature_importance_frame,
    make_candidate_pipeline,
    positive_probabilities,
    slugify,
    source_stratified_group_validation_split,
)
from src.preprocess import DEFAULT_SOURCE_PATHS, prepare_turnover_data


JOB_OUTPUT_DIR = Path("output") / "ml_jobs"
JOB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_OPTIONS: list[str] = list(DEFAULT_SOURCE_PATHS)

SUPPORTED_METRICS: list[str] = [
    "AUC",
    "PR_AUC",
    "F1",
    "Precision",
    "Recall",
    "Balanced_Accuracy",
    "Log_Loss",
    "Brier",
    "Recall@Top10%",
    "Precision@Top10%",
    "Recall@Top20%",
    "Precision@Top20%",
    "Recall@Top30%",
    "Precision@Top30%",
]
DEFAULT_METRICS: list[str] = ["PR_AUC", "AUC", "F1", "Recall@Top20%", "Precision@Top20%"]

METRIC_TOOLTIPS: dict[str, str] = {
    "AUC": "ROC-AUC on the validation/test split; threshold-independent ranking quality.",
    "PR_AUC": "Average precision / PR-AUC on the selected validation/test split.",
    "F1": "F1 at the validation-selected decision threshold.",
    "Precision": "Among predicted leavers, the share that truly left.",
    "Recall": "Among true leavers, the share caught by the model.",
    "Balanced_Accuracy": "Mean of sensitivity and specificity at the selected threshold.",
    "Log_Loss": "Probability calibration loss; lower is better.",
    "Brier": "Mean squared probability error; lower is better.",
    "Recall@Top10%": "Share of leavers captured in the highest-risk 10% of rows.",
    "Precision@Top10%": "Leave rate inside the highest-risk 10% of rows.",
    "Recall@Top20%": "Share of leavers captured in the highest-risk 20% of rows.",
    "Precision@Top20%": "Leave rate inside the highest-risk 20% of rows.",
    "Recall@Top30%": "Share of leavers captured in the highest-risk 30% of rows.",
    "Precision@Top30%": "Leave rate inside the highest-risk 30% of rows.",
}

MODEL_PARAM_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "Logistic Regression": {
        "C": {"label": "Regularization C", "type": "float", "min": 0.01, "max": 10.0, "step": 0.01, "default": 1.0},
        "max_iter": {"label": "Max Iterations", "type": "int", "min": 200, "max": 5000, "step": 100, "default": 2000},
    },
    "Random Forest": {
        "n_estimators": {"label": "Trees", "type": "int", "min": 50, "max": 1000, "step": 25, "default": 300},
        "max_depth": {"label": "Max Depth", "type": "int", "min": 2, "max": 40, "step": 1, "default": 14},
        "min_samples_leaf": {"label": "Min Samples Leaf", "type": "int", "min": 1, "max": 50, "step": 1, "default": 5},
    },
    "XGBoost": {
        "n_estimators": {"label": "Boosting Rounds", "type": "int", "min": 50, "max": 1000, "step": 25, "default": 300},
        "max_depth": {"label": "Max Depth", "type": "int", "min": 2, "max": 16, "step": 1, "default": 4},
        "learning_rate": {"label": "Learning Rate", "type": "float", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.04},
        "min_child_weight": {"label": "Min Child Weight", "type": "float", "min": 0.5, "max": 20.0, "step": 0.5, "default": 3.0},
        "subsample": {"label": "Row Subsample", "type": "float", "min": 0.5, "max": 1.0, "step": 0.05, "default": 0.90},
        "colsample_bytree": {"label": "Column Subsample", "type": "float", "min": 0.5, "max": 1.0, "step": 0.05, "default": 0.85},
        "reg_lambda": {"label": "L2 Regularization", "type": "float", "min": 0.0, "max": 20.0, "step": 0.5, "default": 2.0},
    },
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
for spec in candidate_specs():
    model_prefix = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB"}[spec.model_name]
    payment_label = spec.payment_strategy.replace("_", " ").title()
    model_id = slugify(spec.name)
    MODEL_REGISTRY[model_id] = {
        "label": f"{spec.model_name} - {payment_label}",
        "icon": model_prefix,
        "default_selected": spec.name in {
            "Logistic Regression__learned_imputation",
            "Random Forest__learned_imputation",
            "XGBoost__learned_imputation",
            "XGBoost__native_missing",
            "XGBoost__no_payment",
        },
        "description": f"{spec.model_name} with {spec.payment_strategy.replace('_', ' ')}.",
        "params": MODEL_PARAM_SPECS[spec.model_name],
        "spec": spec,
    }

FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "Tenure": {"prefixes": ["vetek_months", "tenure_years", "career_start_age"], "description": "Tenure, age-at-career-start, and tenure-derived predictors."},
    "Age": {"prefixes": ["age"], "description": "Employee age predictors."},
    "Salary": {"prefixes": ["avg_Payment", "Median_Payment", "stdevp_Payment", "salary", "Sahar"], "description": "Payment level, payment variation, and salary-history predictors."},
    "Workload": {"prefixes": ["avg_omes", "Median_omes", "stdevp_omes", "WorkHours", "omes"], "description": "Workload and workload-history predictors."},
    "Sick Days": {"prefixes": ["avg_illness", "Median_illness", "stdevp_illness", "illness", "hedrut"], "description": "Illness and absence-history predictors."},
    "Manager": {"prefixes": ["manager_Code", "Maneger", "count_managers"], "description": "Manager identity and manager-history predictors."},
    "Role & Dept": {"prefixes": ["contract_type", "tafkid", "Tafkid", "Mahala", "Seif", "Maamad"], "description": "Contract, role, department, rank, and organization context."},
    "Location": {"prefixes": ["Yishuv", "Semel_Yishuv", "distance", "Distance"], "description": "Location and commute-related predictors."},
    "Data Timing": {"prefixes": ["history_", "months_since_previous_record", "elapsed_month"], "description": "Record-history maturity and timing predictors."},
}

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if not isinstance(value, (list, dict, tuple, str)) and pd.isna(value):
        return None
    return value


def _job_snapshot_path(job_id: str) -> Path:
    return JOB_OUTPUT_DIR / f"job_{job_id}.json"


def _summary_path(job_id: str) -> Path:
    return JOB_OUTPUT_DIR / f"summary_{job_id}.json"


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _append_log(job: dict[str, Any], message: str) -> None:
    logs = job.setdefault("logs", [])
    logs.append(f"[{_iso_now()}] {message}")
    if len(logs) > 120:
        job["logs"] = logs[-120:]


def _persist_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        snapshot = json.loads(json.dumps(_json_safe(job)))
    _safe_json_dump(_job_snapshot_path(job_id), snapshot)


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(kwargs)
    _persist_job(job_id)


def allowed_model_ids(train_sources: list[str] | None = None) -> list[str]:
    train_sources = train_sources or ["file1", "file2"]
    ids = []
    for model_id, meta in MODEL_REGISTRY.items():
        spec: CandidateSpec = meta["spec"]
        if "file3" in train_sources and spec.payment_strategy == "learned_imputation":
            continue
        ids.append(model_id)
    return ids


def list_model_options(train_sources: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(allowed_model_ids(train_sources))
    items = []
    for key, meta in MODEL_REGISTRY.items():
        item = {
            "id": key,
            "label": meta["label"],
            "icon": meta["icon"],
            "default_selected": meta["default_selected"] and key in allowed,
            "description": meta.get("description", ""),
            "params": meta.get("params", {}),
            "disabled": key not in allowed,
        }
        if item["disabled"]:
            item["description"] += " Not available when file3 is part of the training sources because learned payment imputation is not allowed to fit on file3."
        items.append(item)
    return items


def metric_tooltip(metric_name: str) -> str:
    return METRIC_TOOLTIPS.get(metric_name, "")


def default_model_ids(train_sources: list[str] | None = None) -> list[str]:
    allowed = set(allowed_model_ids(train_sources))
    return [key for key, meta in MODEL_REGISTRY.items() if meta.get("default_selected") and key in allowed]


def default_hyperparams() -> dict[str, dict[str, Any]]:
    return {
        model_id: {name: spec["default"] for name, spec in meta["params"].items()}
        for model_id, meta in MODEL_REGISTRY.items()
    }


def _sanitize_sources(values: Any, fallback: list[str]) -> list[str]:
    selected = [str(value) for value in values or [] if str(value) in SOURCE_OPTIONS]
    return selected or fallback


def _sanitize_hyperparams(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    defaults = default_hyperparams()
    user_values = config.get("hyperparams", {}) or {}
    sanitized = default_hyperparams()
    for model_id, params in sanitized.items():
        meta = MODEL_REGISTRY[model_id]
        user_model = user_values.get(model_id, {}) or {}
        for name, spec in meta["params"].items():
            raw = user_model.get(name, defaults[model_id][name])
            if spec["type"] == "int":
                value = int(raw)
                value = max(int(spec["min"]), min(int(spec["max"]), value))
            else:
                value = float(raw)
                value = max(float(spec["min"]), min(float(spec["max"]), value))
            params[name] = value
    return sanitized


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    train_sources = _sanitize_sources(config.get("train_sources"), ["file1", "file2"])
    test_sources = _sanitize_sources(config.get("test_sources"), ["file3"])
    allowed = set(allowed_model_ids(train_sources))
    selected_models = [model for model in config.get("selected_models", []) if model in allowed]
    if not selected_models:
        selected_models = default_model_ids(train_sources)

    selected_metrics = [metric for metric in config.get("selected_metrics", []) if metric in SUPPORTED_METRICS]
    if not selected_metrics:
        selected_metrics = list(DEFAULT_METRICS)

    ranking_metric = config.get("ranking_metric", "PR_AUC")
    if ranking_metric not in selected_metrics:
        ranking_metric = selected_metrics[0]

    validation_size = float(config.get("val_size", 0.20))
    validation_size = max(0.10, min(0.40, validation_size))
    drop_feature_groups = [group for group in config.get("drop_feature_groups", []) if group in FEATURE_GROUPS]

    return {
        "selected_models": selected_models,
        "selected_metrics": selected_metrics,
        "ranking_metric": ranking_metric,
        "random_seed": int(config.get("random_seed", 42)),
        "val_size": validation_size,
        "train_sources": train_sources,
        "test_sources": test_sources,
        "drop_feature_groups": drop_feature_groups,
        "workflow": f"{', '.join(train_sources)} train/validation -> {', '.join(test_sources)} test",
        "hyperparams": _sanitize_hyperparams(config),
    }


def _columns_to_drop(feature_groups: list[str], all_columns: list[str]) -> list[str]:
    prefixes: list[str] = []
    for group in feature_groups:
        prefixes.extend(FEATURE_GROUPS.get(group, {}).get("prefixes", []))
    if not prefixes:
        return []
    return [
        column for column in all_columns
        if any(column == prefix or column.startswith(prefix) or column.startswith(f"{prefix}_") for prefix in prefixes)
    ]


def _metric_value(row: dict[str, Any], metric: str, prefix: str = "") -> float:
    value = row.get(f"{prefix}{metric}")
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _prediction_export(prepared, probabilities: np.ndarray, threshold: float, job_id: str) -> tuple[str, int, dict[str, int]]:
    columns = [
        column for column in ["source", "source_employee_id", "fictive_employee", "calc_month", "leave_ind", "age", "vetek_months", "contract_type"]
        if column in prepared.test_frame.columns
    ]
    output = prepared.test_frame[columns].copy()
    output["turnover_probability"] = probabilities
    output["turnover_prediction"] = (probabilities >= threshold).astype(int)
    output["risk_rank"] = output["turnover_probability"].rank(method="first", ascending=False).astype(int)
    output["risk_category"] = pd.cut(
        output["turnover_probability"],
        bins=[0.0, 0.30, 0.50, 0.70, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
        include_lowest=True,
    )
    output = output.sort_values("turnover_probability", ascending=False)
    path = JOB_OUTPUT_DIR / f"predictions_{job_id}.xlsx"
    output.to_excel(path, index=False)
    risk_dist = {str(key): int(value) for key, value in output["risk_category"].value_counts().items()}
    return str(path), len(output), risk_dist


def _save_artifact(spec: CandidateSpec, pipeline, columns: list[str], threshold: float, metrics: dict[str, Any], config: dict[str, Any], job_id: str) -> str:
    artifact = {
        "artifact_version": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": "ml_workbench",
        "job_id": job_id,
        "candidate": spec.name,
        "model_name": spec.model_name,
        "payment_strategy": spec.payment_strategy,
        "pipeline": pipeline,
        "feature_columns": columns,
        "decision_threshold": threshold,
        "target_column": "leave_ind",
        "train_sources": config["train_sources"],
        "test_sources": config["test_sources"],
        "hyperparams": config["hyperparams"].get(slugify(spec.name), {}),
        "validation_metrics": {key: value for key, value in metrics.items() if key.startswith("Val_")},
        "test_metrics": {key: value for key, value in metrics.items() if key.startswith("Test_")},
    }
    path = ARTIFACT_DIR / f"ml_workbench_{slugify(spec.name)}_{job_id}.pkl"
    joblib.dump(artifact, path)
    return str(path)


def _write_results_report(job_id: str, config: dict[str, Any], results_df: pd.DataFrame, best_model_name: str, split_info: dict[str, Any]) -> str:
    path = JOB_OUTPUT_DIR / f"results_{job_id}.txt"
    lines = [
        "ML WORKBENCH - TURNOVER EXPERIMENT",
        "=" * 80,
        f"Job ID: {job_id}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Workflow: {config['workflow']}",
        f"Ranking metric: validation {config['ranking_metric']}",
        f"Best model: {best_model_name}",
        "",
        "Split summary",
        f"- Train sources: {', '.join(config['train_sources'])}",
        f"- Test sources: {', '.join(config['test_sources'])}",
        f"- Inner train rows: {split_info.get('train')}",
        f"- Validation rows: {split_info.get('val')}",
        f"- Test rows: {split_info.get('test')}",
        "",
        "Model comparison",
        results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
    config = job["config"]

    try:
        _update_job(job_id, status="running", stage="data_analysis", stage_label="Preparing Data", progress=5, started_at=_iso_now())
        with _jobs_lock:
            _append_log(_jobs[job_id], f"Preparing data for {config['workflow']}")
        _persist_job(job_id)

        prepared = prepare_turnover_data(train_sources=config["train_sources"], test_sources=config["test_sources"])
        train_mask, validation_mask = source_stratified_group_validation_split(
            prepared.X_train,
            prepared.y_train,
            prepared.train_groups,
            validation_size=config["val_size"],
            seed=config["random_seed"],
        )
        X_inner = prepared.X_train.loc[train_mask]
        y_inner = prepared.y_train.loc[train_mask]
        X_validation = prepared.X_train.loc[validation_mask]
        y_validation = prepared.y_train.loc[validation_mask]

        split_info = {
            "train": int(len(X_inner)),
            "val": int(len(X_validation)),
            "test": int(len(prepared.X_test)),
            "total": int(len(prepared.X_train) + len(prepared.X_test)),
            "train_sources": " + ".join(config["train_sources"]),
            "test_source": " + ".join(config["test_sources"]),
        }
        _update_job(job_id, progress=20, split_info=split_info)

        selected_models = config["selected_models"]
        total_models = len(selected_models)
        all_results: list[dict[str, Any]] = []
        trained: dict[str, dict[str, Any]] = {}
        best_model_id = ""
        best_model_name = ""
        best_score = -np.inf

        _update_job(job_id, stage="training", stage_label="Training Candidates", progress=30)
        for index, model_id in enumerate(selected_models, start=1):
            meta = MODEL_REGISTRY[model_id]
            spec: CandidateSpec = meta["spec"]
            model_label = meta["label"]
            model_params = config["hyperparams"].get(model_id, {})
            with _jobs_lock:
                _jobs[job_id]["active_model"] = model_id
                _append_log(_jobs[job_id], f"Validating {model_label}")
            _persist_job(job_id)

            columns = candidate_feature_columns(X_inner, spec)
            dropped_columns = _columns_to_drop(config["drop_feature_groups"], columns)
            if dropped_columns:
                columns = [column for column in columns if column not in set(dropped_columns)]
            if not columns:
                raise ValueError(f"No columns left for {model_label} after feature-group exclusions.")

            validation_pipeline = make_candidate_pipeline(
                spec, X_inner[columns], y_inner, config["random_seed"], estimator_params=model_params
            )
            with contextlib.redirect_stdout(io.StringIO()):
                validation_pipeline.fit(X_inner[columns], y_inner)
            val_probabilities = positive_probabilities(validation_pipeline, X_validation[columns])
            threshold = best_f1_threshold(y_validation, val_probabilities)

            final_columns = candidate_feature_columns(prepared.X_train, spec)
            if dropped_columns:
                final_columns = [column for column in final_columns if column not in set(dropped_columns)]
            final_pipeline = make_candidate_pipeline(
                spec, prepared.X_train[final_columns], prepared.y_train, config["random_seed"], estimator_params=model_params
            )
            with contextlib.redirect_stdout(io.StringIO()):
                final_pipeline.fit(prepared.X_train[final_columns], prepared.y_train)

            train_probabilities = positive_probabilities(final_pipeline, prepared.X_train[final_columns])
            test_probabilities = positive_probabilities(final_pipeline, prepared.X_test[final_columns])
            train_metrics = evaluate_probabilities(prepared.y_train, train_probabilities, threshold)
            val_metrics = evaluate_probabilities(y_validation, val_probabilities, threshold)
            test_metrics = evaluate_probabilities(prepared.y_test, test_probabilities, threshold)

            result_row: dict[str, Any] = {
                "model_id": model_id,
                "Model": model_label,
                "Candidate": spec.name,
                "Validation_Threshold": threshold,
                "Feature_Count": len(final_columns),
                "Dropped_Features": len(dropped_columns),
            }
            for metric_name in SUPPORTED_METRICS:
                if metric_name in train_metrics:
                    result_row[f"Train_{metric_name}"] = float(train_metrics[metric_name])
                if metric_name in val_metrics:
                    result_row[f"Val_{metric_name}"] = float(val_metrics[metric_name])
                if metric_name in test_metrics:
                    result_row[metric_name] = float(test_metrics[metric_name])

            ranking_metric = config["ranking_metric"]
            ranking_value = _metric_value(result_row, ranking_metric, prefix="Val_")
            result_row["Ranking_Value"] = ranking_value
            all_results.append(result_row)
            trained[model_id] = {
                "spec": spec,
                "pipeline": final_pipeline,
                "columns": final_columns,
                "threshold": threshold,
                "test_probabilities": test_probabilities,
                "metrics": result_row,
            }

            if np.isfinite(ranking_value) and ranking_value > best_score:
                best_score = ranking_value
                best_model_id = model_id
                best_model_name = model_label

            with _jobs_lock:
                _jobs[job_id].setdefault("completed_models", []).append(model_id)
            _update_job(job_id, progress=30 + int((index / total_models) * 45))

        if not best_model_id:
            raise ValueError("No model produced a finite validation score for the selected ranking metric.")

        _update_job(job_id, stage="evaluation", stage_label="Evaluating Test Sources", progress=82, active_model=None)
        results_df = pd.DataFrame(all_results).sort_values("Ranking_Value", ascending=False).reset_index(drop=True)

        best = trained[best_model_id]
        importance = feature_importance_frame(best["pipeline"], best["spec"].model_name).head(15)
        importance_json = [
            {"feature": str(row["Feature"]), "score": round(float(row["Importance"]), 6), "description": str(row["Feature"])}
            for _, row in importance.iterrows()
        ]

        predictions_path, n_employees, risk_dist = _prediction_export(prepared, best["test_probabilities"], best["threshold"], job_id)
        artifact_path = _save_artifact(best["spec"], best["pipeline"], best["columns"], best["threshold"], best["metrics"], config, job_id)

        selected_cols = ["Model", "Validation_Threshold", "Feature_Count"]
        for metric in config["selected_metrics"]:
            selected_cols.extend([f"Train_{metric}", f"Val_{metric}", metric])
        selected_cols = [column for column in selected_cols if column in results_df.columns]
        comparison_df = results_df[selected_cols].copy()

        results_text_path = _write_results_report(job_id, config, comparison_df, best_model_name, split_info)
        summary = {
            "job_id": job_id,
            "completed_at": _iso_now(),
            "config": config,
            "best_model_id": best_model_id,
            "best_model_name": best_model_name,
            "ranking_metric": config["ranking_metric"],
            "ranking_value": best_score,
            "rows": split_info["total"],
            "split": split_info,
            "results": comparison_df.to_dict(orient="records"),
            "cleaning_audit": asdict(prepared.cleaning_audit),
        }
        summary_path = _summary_path(job_id)
        _safe_json_dump(summary_path, summary)

        with _jobs_lock:
            current = _jobs[job_id]
            current["results"] = summary["results"]
            current["summary_path"] = str(summary_path)
            current["feature_importance"] = importance_json
            current["feature_importance_method"] = "native transformed-feature importance"
            current["predictions_path"] = predictions_path
            current["artifact_path"] = artifact_path
            current["results_text_path"] = results_text_path
            current["risk_distribution"] = risk_dist
            current["uncertainty"] = None
            current["n_predictions"] = n_employees
            _append_log(current, f"Best model: {best_model_name} (validation {config['ranking_metric']}={best_score:.4f})")
            _append_log(current, f"Artifact saved: {artifact_path}")
            _append_log(current, f"Predictions saved: {predictions_path}")
        _update_job(job_id, status="completed", stage="results_ready", stage_label="Results Ready", progress=100, best_model=best_model_name, finished_at=_iso_now())
    except Exception as exc:
        with _jobs_lock:
            _append_log(_jobs[job_id], f"FAILED: {exc}")
        _update_job(job_id, status="failed", stage="failed", stage_label="Failed", finished_at=_iso_now(), error=str(exc))


def start_training_job(config: dict[str, Any], data_path: str | None = None, split_dir: str | None = None) -> str:
    safe_config = sanitize_config(config)
    job_id = uuid.uuid4().hex[:10]
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "stage_label": "Queued",
        "progress": 0,
        "active_model": None,
        "created_at": _iso_now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "results": [],
        "summary_path": None,
        "logs": [f"[{_iso_now()}] Job created"],
        "config": safe_config,
        "data_path": safe_config["workflow"],
        "split_dir": None,
        "completed_models": [],
        "split_info": {},
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _persist_job(job_id)
    _executor.submit(_run_job, job_id)
    return job_id


def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        in_memory = _jobs.get(job_id)
        if in_memory is not None:
            return json.loads(json.dumps(_json_safe(in_memory)))
    snapshot = _job_snapshot_path(job_id)
    if snapshot.exists():
        return json.loads(snapshot.read_text(encoding="utf-8"))
    return {}


def list_jobs(limit: int = 10) -> list[dict[str, Any]]:
    with _jobs_lock:
        jobs = list(_jobs.values())
    if len(jobs) < limit:
        for path in sorted(JOB_OUTPUT_DIR.glob("job_*.json"), reverse=True):
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not any(job.get("job_id") == loaded.get("job_id") for job in jobs):
                jobs.append(loaded)
    jobs = sorted(jobs, key=lambda job: job.get("created_at", ""), reverse=True)
    return [json.loads(json.dumps(_json_safe(job))) for job in jobs[:limit]]
