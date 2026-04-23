import contextlib
import io
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET_COL, set_seed
from src.data_loader import RealExcelDataLoader
from src.evaluator import Evaluator
from src.models.classifiers import (
    AdaBoostTurnover,
    EnsembleTurnover,
    LogisticRegressionTurnover,
    RandomForestTurnover,
    XGBoostTurnover,
)
from src.models.nn_advanced import RegularizedMLPTurnover
from src.models.nn_model import NeuralNetTurnover

JOB_OUTPUT_DIR = os.path.join("output", "ml_jobs")
os.makedirs(JOB_OUTPUT_DIR, exist_ok=True)


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "icon": "📈",
        "default_selected": True,
        "description": "Linear probabilistic baseline. Computes P(leave=1) with a sigmoid over weighted features; strong for interpretable coefficients and fast training.",
        "params": {
            "max_iter": {"label": "Max Iterations", "type": "int", "min": 200, "max": 5000, "step": 50, "default": 1000},
            "C": {"label": "Regularization C", "type": "float", "min": 0.01, "max": 10.0, "step": 0.01, "default": 1.0},
        },
    },
    "random_forest": {
        "label": "Random Forest",
        "icon": "🌲",
        "default_selected": True,
        "description": "Bagging ensemble of decision trees trained on random feature subsets. Final probability is the average tree vote, reducing variance and overfitting.",
        "params": {
            "n_estimators": {"label": "Trees", "type": "int", "min": 50, "max": 1000, "step": 25, "default": 200},
            "max_depth": {"label": "Max Depth", "type": "int", "min": 2, "max": 40, "step": 1, "default": 15},
            "min_samples_split": {"label": "Min Samples Split", "type": "int", "min": 2, "max": 30, "step": 1, "default": 2},
        },
    },
    "xgboost": {
        "label": "XGBoost",
        "icon": "⚡",
        "default_selected": True,
        "description": "Gradient-boosted trees added sequentially to minimize loss. Strong nonlinear performance and robust handling of tabular feature interactions.",
        "params": {
            "n_estimators": {"label": "Boosting Rounds", "type": "int", "min": 50, "max": 1000, "step": 25, "default": 200},
            "max_depth": {"label": "Max Depth", "type": "int", "min": 2, "max": 16, "step": 1, "default": 6},
            "learning_rate": {"label": "Learning Rate", "type": "float", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.1},
            "subsample": {"label": "Row Subsample", "type": "float", "min": 0.5, "max": 1.0, "step": 0.05, "default": 1.0},
            "colsample_bytree": {"label": "Column Subsample", "type": "float", "min": 0.5, "max": 1.0, "step": 0.05, "default": 1.0},
        },
    },
    "adaboost": {
        "label": "AdaBoost",
        "icon": "🧠",
        "default_selected": True,
        "description": "Sequential boosting that emphasizes previously misclassified samples. Combines weak learners into a stronger classifier.",
        "params": {
            "n_estimators": {"label": "Estimators", "type": "int", "min": 50, "max": 1000, "step": 25, "default": 200},
            "learning_rate": {"label": "Learning Rate", "type": "float", "min": 0.01, "max": 2.0, "step": 0.01, "default": 0.5},
        },
    },
    "ensemble": {
        "label": "Voting Ensemble",
        "icon": "🧩",
        "default_selected": True,
        "description": "Soft-voting meta-model that averages probabilities from multiple base learners to improve robustness across scenarios.",
        "params": {},
    },
    "neural_net": {
        "label": "Neural Net",
        "icon": "🤖",
        "default_selected": True,
        "description": "Feedforward neural network with batch normalization and dropout. Learns nonlinear patterns via gradient descent.",
        "params": {
            "epochs": {"label": "Epochs", "type": "int", "min": 10, "max": 400, "step": 5, "default": 100},
            "batch_size": {"label": "Batch Size", "type": "int", "min": 8, "max": 256, "step": 8, "default": 32},
            "learning_rate": {"label": "Learning Rate", "type": "float", "min": 0.0001, "max": 0.02, "step": 0.0001, "default": 0.001},
        },
    },
    "deep_mlp": {
        "label": "Deep MLP",
        "icon": "🧪",
        "default_selected": True,
        "description": "Regularized deep multilayer perceptron with residual blocks and cross-validation folds for stronger generalization.",
        "params": {
            "epochs": {"label": "Epochs", "type": "int", "min": 20, "max": 500, "step": 10, "default": 150},
            "batch_size": {"label": "Batch Size", "type": "int", "min": 8, "max": 256, "step": 8, "default": 32},
            "lr": {"label": "Learning Rate", "type": "float", "min": 0.0001, "max": 0.02, "step": 0.0001, "default": 0.001},
            "n_folds": {"label": "CV Folds", "type": "int", "min": 2, "max": 10, "step": 1, "default": 5},
        },
    },
}

DEFAULT_METRICS: List[str] = ["AUC_ROC", "F1_Score", "Precision", "Recall", "Recall@Top20%", "Precision@Top20%"]
SUPPORTED_METRICS: List[str] = [
    "AUC_ROC",
    "F1_Score",
    "Precision",
    "Recall",
    "Recall@Top20%",
    "Max_Recall@Top20%",
    "Precision@Top20%",
    "Recall@Top50%",
    "Primary_Probability",
    "Avg_Error_Cost",
    "Error_Cost",
    "Improvement_Factor",
]

METRIC_TOOLTIPS: Dict[str, str] = {
    "AUC_ROC": "Area under the ROC curve; threshold-independent ranking quality. Higher means positives are ranked above negatives more often.",
    "F1_Score": "Harmonic mean of precision and recall: F1 = 2PR/(P+R). Useful when both false positives and false negatives matter.",
    "Precision": "Among predicted leavers, how many truly left: TP/(TP+FP). Higher precision means fewer false alarms.",
    "Recall": "Among true leavers, how many the model catches: TP/(TP+FN). Higher recall means fewer missed leavers.",
    "Recall@Top20%": "Sort employees by predicted risk and inspect top 20%. This is captured leavers divided by total leavers.",
    "Max_Recall@Top20%": "Theoretical recall ceiling at top 20% given class counts; upper bound for Recall@Top20%.",
    "Precision@Top20%": "Hit rate inside the highest-risk 20% bucket: true leavers in top 20% divided by top-20% size.",
    "Recall@Top50%": "Coverage of leavers when acting on the top half of ranked risk scores.",
    "Primary_Probability": "Base leave rate in the evaluated sample (positive class prevalence).",
    "Avg_Error_Cost": "Model log-loss averaged over samples using predicted probabilities.",
    "Error_Cost": "Baseline log-loss from always predicting the base rate.",
    "Improvement_Factor": "Baseline error cost divided by model error cost; >1 indicates improvement over no-skill baseline.",
}


_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _job_snapshot_path(job_id: str) -> str:
    return os.path.join(JOB_OUTPUT_DIR, f"job_{job_id}.json")


def _summary_path(job_id: str) -> str:
    return os.path.join(JOB_OUTPUT_DIR, f"summary_{job_id}.json")


def _safe_json_dump(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _append_log(job: Dict[str, Any], message: str) -> None:
    logs = job.setdefault("logs", [])
    logs.append(f"[{_iso_now()}] {message}")
    if len(logs) > 120:
        job["logs"] = logs[-120:]


def _persist_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        _safe_json_dump(_job_snapshot_path(job_id), job)


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(kwargs)
    _persist_job(job_id)


def list_model_options() -> List[Dict[str, Any]]:
    items = []
    for key, meta in MODEL_REGISTRY.items():
        items.append(
            {
                "id": key,
                "label": meta["label"],
                "icon": meta["icon"],
                "default_selected": meta["default_selected"],
                "description": meta.get("description", ""),
                "params": meta["params"],
            }
        )
    return items


def metric_tooltip(metric_name: str) -> str:
    return METRIC_TOOLTIPS.get(metric_name, "")


def default_model_ids() -> List[str]:
    return [k for k, v in MODEL_REGISTRY.items() if v.get("default_selected")]


def default_hyperparams() -> Dict[str, Dict[str, Any]]:
    params: Dict[str, Dict[str, Any]] = {}
    for model_id, meta in MODEL_REGISTRY.items():
        params[model_id] = {name: spec["default"] for name, spec in meta["params"].items()}
    return params


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    selected_models = [m for m in config.get("selected_models", []) if m in MODEL_REGISTRY]
    if not selected_models:
        selected_models = default_model_ids()

    selected_metrics = [m for m in config.get("selected_metrics", []) if m in SUPPORTED_METRICS]
    if not selected_metrics:
        selected_metrics = list(DEFAULT_METRICS)

    threshold = float(config.get("threshold", 0.5))
    threshold = max(0.05, min(0.95, threshold))

    random_seed = int(config.get("random_seed", 42))

    split_mode = config.get("split_mode", "random")
    if split_mode not in {"random", "fixed"}:
        split_mode = "random"

    ratio = float(config.get("test_size", 0.2))
    ratio = max(0.1, min(0.4, ratio))

    val_ratio = float(config.get("val_size", 0.25))
    val_ratio = max(0.1, min(0.4, val_ratio))

    hyperparams = default_hyperparams()
    user_hyperparams = config.get("hyperparams", {})

    for model_id, params in hyperparams.items():
        model_spec = MODEL_REGISTRY[model_id]["params"]
        user_model_params = user_hyperparams.get(model_id, {})
        for param_name, spec in model_spec.items():
            user_value = user_model_params.get(param_name, params[param_name])
            if spec["type"] == "int":
                safe_value = int(user_value)
                safe_value = max(int(spec["min"]), min(int(spec["max"]), safe_value))
            else:
                safe_value = float(user_value)
                safe_value = max(float(spec["min"]), min(float(spec["max"]), safe_value))
            params[param_name] = safe_value

    ranking_metric = config.get("ranking_metric", "AUC_ROC")
    if ranking_metric not in selected_metrics:
        ranking_metric = selected_metrics[0]

    return {
        "selected_models": selected_models,
        "selected_metrics": selected_metrics,
        "threshold": threshold,
        "random_seed": random_seed,
        "split_mode": split_mode,
        "test_size": ratio,
        "val_size": val_ratio,
        "ranking_metric": ranking_metric,
        "hyperparams": hyperparams,
    }


def _build_model(model_id: str, params: Dict[str, Any], scale_pos_weight: float):
    if model_id == "logistic_regression":
        return LogisticRegressionTurnover(class_weight="balanced", max_iter=int(params.get("max_iter", 1000)), C=float(params.get("C", 1.0)))
    if model_id == "random_forest":
        return RandomForestTurnover(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 15)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            class_weight="balanced",
        )
    if model_id == "xgboost":
        return XGBoostTurnover(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        )
    if model_id == "adaboost":
        return AdaBoostTurnover(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.5)),
        )
    if model_id == "ensemble":
        return EnsembleTurnover()
    if model_id == "neural_net":
        return NeuralNetTurnover(
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
        )
    if model_id == "deep_mlp":
        return RegularizedMLPTurnover(
            epochs=int(params.get("epochs", 150)),
            batch_size=int(params.get("batch_size", 32)),
            lr=float(params.get("lr", 0.001)),
            n_folds=int(params.get("n_folds", 5)),
        )
    raise ValueError(f"Unsupported model id: {model_id}")


def _resolve_dataset_config(data_path: str) -> Dict[str, Any]:
    dataset_tag = os.path.splitext(os.path.basename(data_path))[0]
    if dataset_tag == "factory_two":
        return {"employee_id_col": "fictive-oved", "time_col": None}
    return {"employee_id_col": "fictive2", "time_col": "fictive-ovedmiun"}


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
    config = job["config"]

    try:
        _update_job(job_id, status="running", stage="data_analysis", stage_label="Analyzing Data", progress=5, started_at=_iso_now())
        with _jobs_lock:
            _append_log(_jobs[job_id], "Loading and preprocessing data")
        _persist_job(job_id)

        set_seed(config["random_seed"])

        data_path = job["data_path"]
        split_dir = job.get("split_dir")
        dataset_config = _resolve_dataset_config(data_path)

        loader = RealExcelDataLoader(data_path, **dataset_config)
        raw_df = loader.load()
        processed_df = loader.preprocess(raw_df)

        if TARGET_COL not in processed_df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in preprocessed data")

        X = processed_df.drop(columns=[TARGET_COL])
        y = processed_df[TARGET_COL]
        _update_job(job_id, progress=20)

        if config["split_mode"] == "fixed":
            if not split_dir:
                raise ValueError("Fixed split mode selected but split directory is missing")
            train_ids_path = os.path.join(split_dir, "train_ids.txt")
            test_ids_path = os.path.join(split_dir, "test_ids.txt")
            if not (os.path.exists(train_ids_path) and os.path.exists(test_ids_path)):
                raise ValueError("train_ids.txt or test_ids.txt not found in split directory")

            with open(train_ids_path, "r", encoding="utf-8") as f:
                train_ids = {line.strip() for line in f if line.strip()}
            with open(test_ids_path, "r", encoding="utf-8") as f:
                test_ids = {line.strip() for line in f if line.strip()}

            kept_ids = [str(v).strip() for v in loader.get_kept_indices()]
            train_mask = [eid in train_ids for eid in kept_ids]
            test_mask = [eid in test_ids for eid in kept_ids]

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            X_val = X_test
            y_val = y_test
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=config["test_size"],
                random_state=config["random_seed"],
                stratify=y,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=config["val_size"],
                random_state=config["random_seed"],
                stratify=y_temp,
            )

        _update_job(job_id, stage="training", stage_label="Training Models", progress=30)
        with _jobs_lock:
            _append_log(_jobs[job_id], f"Split sizes - train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
        _persist_job(job_id)

        neg_count = max(float((y_train == 0).sum()), 1.0)
        pos_count = max(float((y_train == 1).sum()), 1.0)
        scale_pos_weight = neg_count / pos_count

        evaluator = Evaluator()
        all_results = []
        best_model_id = None
        best_model_name = ""
        best_score = -np.inf

        selected_models = config["selected_models"]
        total_models = len(selected_models)

        for index, model_id in enumerate(selected_models, start=1):
            model_label = MODEL_REGISTRY[model_id]["label"]
            with _jobs_lock:
                _jobs[job_id]["active_model"] = model_id
                _append_log(_jobs[job_id], f"Training {model_label}")
            _persist_job(job_id)

            model = _build_model(model_id, config["hyperparams"].get(model_id, {}), scale_pos_weight)
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X_train, y_train)

            val_metrics = evaluator.evaluate(model, X_val, y_val, threshold=config["threshold"])
            test_metrics = evaluator.evaluate(model, X_test, y_test, threshold=config["threshold"])

            result_row = {"model_id": model_id, "Model": model_label}
            for metric_name in SUPPORTED_METRICS:
                if metric_name in test_metrics:
                    result_row[metric_name] = float(test_metrics[metric_name])
                if metric_name in val_metrics:
                    result_row[f"Val_{metric_name}"] = float(val_metrics[metric_name])

            ranking_metric = config["ranking_metric"]
            ranking_value = float(val_metrics.get(ranking_metric, test_metrics.get(ranking_metric, -np.inf)))
            result_row["Ranking_Value"] = ranking_value
            all_results.append(result_row)

            if ranking_value > best_score:
                best_score = ranking_value
                best_model_id = model_id
                best_model_name = model_label

            progress = 30 + int((index / total_models) * 45)
            _update_job(job_id, progress=progress)

        _update_job(job_id, stage="validation", stage_label="Validation", progress=78, active_model=None)
        _update_job(job_id, stage="evaluation", stage_label="Evaluating Results", progress=88)

        results_df = pd.DataFrame(all_results).sort_values(by="Ranking_Value", ascending=False)
        selected_cols = ["Model"]
        for metric_name in config["selected_metrics"]:
            val_col = f"Val_{metric_name}"
            if val_col in results_df.columns:
                selected_cols.append(val_col)
            if metric_name in results_df.columns:
                selected_cols.append(metric_name)

        comparison_df = results_df[selected_cols].copy()
        summary = {
            "job_id": job_id,
            "completed_at": _iso_now(),
            "config": config,
            "best_model_id": best_model_id,
            "best_model_name": best_model_name,
            "ranking_metric": config["ranking_metric"],
            "ranking_value": float(best_score) if np.isfinite(best_score) else None,
            "rows": int(len(X)),
            "split": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
            "results": comparison_df.to_dict(orient="records"),
        }

        summary_path = _summary_path(job_id)
        _safe_json_dump(summary_path, summary)

        with _jobs_lock:
            _jobs[job_id]["results"] = summary["results"]
            _jobs[job_id]["summary_path"] = summary_path
            _append_log(_jobs[job_id], f"Best model: {best_model_name} ({config['ranking_metric']}={best_score:.4f})")
            _append_log(_jobs[job_id], f"Summary saved to {summary_path}")
        _update_job(
            job_id,
            status="completed",
            stage="results_ready",
            stage_label="Results Ready",
            progress=100,
            best_model=best_model_name,
            finished_at=_iso_now(),
        )
    except Exception as exc:
        with _jobs_lock:
            _append_log(_jobs[job_id], f"FAILED: {exc}")
        _update_job(job_id, status="failed", stage="failed", stage_label="Failed", finished_at=_iso_now(), error=str(exc))


def start_training_job(config: Dict[str, Any], data_path: str, split_dir: str = "split") -> str:
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
        "data_path": data_path,
        "split_dir": split_dir,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _persist_job(job_id)
    _executor.submit(_run_job, job_id)
    return job_id


def get_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        in_mem = _jobs.get(job_id)
        if in_mem is not None:
            return json.loads(json.dumps(in_mem))

    snapshot_path = _job_snapshot_path(job_id)
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def list_jobs(limit: int = 10) -> List[Dict[str, Any]]:
    with _jobs_lock:
        jobs = list(_jobs.values())
    jobs = sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)
    return [json.loads(json.dumps(j)) for j in jobs[:limit]]
