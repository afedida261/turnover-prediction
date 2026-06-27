from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import TARGET_COL
from src.datasets import DatasetSpec, read_excel_with_header_detection, spec_for_path
from src.static_preprocessing import (
    EXIT_METADATA_COLUMNS,
    build_static_model_frame,
    make_tabular_preprocessor,
    split_X_y,
)


FILEX_PATHS = [Path("data/file1.xlsx"), Path("data/file2.xlsx"), Path("data/file3.xlsx")]


def clean_X_for_model(X: pd.DataFrame, employee_id_col: str) -> pd.DataFrame:
    return X.drop(columns=[employee_id_col], errors="ignore")


def split_data(X, y, seed, groups=None):
    if y.nunique() < 2:
        raise ValueError("Need both target classes.")

    if groups is None:
        stratify = y if y.value_counts().min() >= 3 else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )
        stratify_temp = y_temp if y_temp.value_counts().min() >= 3 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            random_state=seed,
            stratify=stratify_temp,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    groups = pd.Series(groups, index=X.index).astype(str)
    group_labels = (
        pd.DataFrame({"group": groups, TARGET_COL: y})
        .groupby("group", sort=False)[TARGET_COL]
        .max()
    )
    group_ids = group_labels.index.to_numpy()
    label_values = group_labels.to_numpy()
    stratify = label_values if pd.Series(label_values).value_counts().min() >= 2 else None

    train_val_ids, test_ids = train_test_split(
        group_ids,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    train_val_labels = group_labels.loc[train_val_ids].to_numpy()
    stratify_train_val = (
        train_val_labels
        if pd.Series(train_val_labels).value_counts().min() >= 2
        else None
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.25,
        random_state=seed,
        stratify=stratify_train_val,
    )

    train_mask = groups.isin(train_ids)
    val_mask = groups.isin(val_ids)
    test_mask = groups.isin(test_ids)
    return (
        X.loc[train_mask],
        X.loc[val_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[val_mask],
        y.loc[test_mask],
    )


def split_train_val(X, y, seed, groups=None, val_size=0.2):
    if y.nunique() < 2:
        raise ValueError("Need both target classes.")

    if groups is None:
        stratify = y if y.value_counts().min() >= 2 else None
        return train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=seed,
            stratify=stratify,
        )

    groups = pd.Series(groups, index=X.index).astype(str)
    group_labels = (
        pd.DataFrame({"group": groups, TARGET_COL: y})
        .groupby("group", sort=False)[TARGET_COL]
        .max()
    )
    group_ids = group_labels.index.to_numpy()
    label_values = group_labels.to_numpy()
    stratify = label_values if pd.Series(label_values).value_counts().min() >= 2 else None
    train_ids, val_ids = train_test_split(
        group_ids,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    train_mask = groups.isin(train_ids)
    val_mask = groups.isin(val_ids)
    return X.loc[train_mask], X.loc[val_mask], y.loc[train_mask], y.loc[val_mask]


def model_builders(y, seed):
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    scale_pos_weight = negatives / positives

    return {
        "Logistic Regression": lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
        ),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=180,
            learning_rate=0.5,
            random_state=seed,
        ),
        "Ensemble": lambda: VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=seed,
                    ),
                ),
                ("ada", AdaBoostClassifier(n_estimators=180, learning_rate=0.5, random_state=seed)),
            ],
            voting="soft",
        ),
    }


def class_weight_model_builders(y, seed):
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    scale_pos_weight = negatives / positives

    return {
        "Logistic Regression": lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
        ),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=180,
            learning_rate=0.5,
            random_state=seed,
        ),
        "Ensemble": lambda: VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=seed,
                    ),
                ),
                (
                    "ada",
                    AdaBoostClassifier(
                        n_estimators=180,
                        learning_rate=0.5,
                        random_state=seed,
                    ),
                ),
            ],
            voting="soft",
        ),
    }


def ipw_model_builders(y, seed):
    del y
    return {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=seed,
            n_jobs=1,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        ),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=180,
            learning_rate=0.5,
            random_state=seed,
        ),
        "Ensemble": lambda: VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=seed,
                    ),
                ),
                ("ada", AdaBoostClassifier(n_estimators=180, learning_rate=0.5, random_state=seed)),
            ],
            voting="soft",
        ),
    }


def inverse_probability_weights(y: pd.Series) -> np.ndarray:
    y = pd.Series(y).astype(int)
    prevalence = y.value_counts(normalize=True).to_dict()
    return y.map(lambda value: 0.5 / prevalence.get(value, 1.0)).to_numpy(dtype=float)


def cumulative_unique_count(values: pd.Series) -> pd.Series:
    seen = set()
    counts = []
    for value in values.astype("string"):
        if pd.notna(value):
            seen.add(str(value))
        counts.append(len(seen))
    return pd.Series(counts, index=values.index)


def top_k_metrics(y_true, y_score, fractions=(0.1, 0.2, 0.3)) -> dict[str, float]:
    ranked = pd.DataFrame({"y_true": np.asarray(y_true), "score": np.asarray(y_score)})
    ranked = ranked.sort_values("score", ascending=False)
    total_positive = float(ranked["y_true"].sum())
    metrics = {}
    for frac in fractions:
        top_n = max(1, int(np.ceil(len(ranked) * frac)))
        segment = ranked.head(top_n)
        captured = float(segment["y_true"].sum())
        recall = captured / total_positive if total_positive > 0 else 0.0
        precision = captured / top_n if top_n > 0 else 0.0
        label = int(frac * 100)
        metrics[f"Recall@Top{label}%"] = recall
        metrics[f"Precision@Top{label}%"] = precision
    return metrics


def evaluate_pipeline(pipe, X, y) -> dict[str, float]:
    probs = pipe.predict_proba(X)
    if probs.ndim == 2:
        probs = probs[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = np.nan if y.nunique() < 2 else roc_auc_score(y, probs)
    metrics = {
        "AUC": auc,
        "F1": f1_score(y, preds, zero_division=0),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall": recall_score(y, preds, zero_division=0),
    }
    metrics.update(top_k_metrics(y, probs))
    return metrics


def build_pipeline(X_train: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_tabular_preprocessor(X_train)),
            ("model", estimator),
        ]
    )


def load_experiment_frame(spec: DatasetSpec, *, drop_exit_metadata: bool = True) -> dict[str, Any]:
    raw_df, header_row = read_excel_with_header_detection(spec.path)
    model_frame, dropped_columns = build_static_model_frame(
        raw_df,
        spec,
        drop_exit_metadata=drop_exit_metadata,
    )
    model_frame = model_frame.dropna(subset=[TARGET_COL]).copy()
    model_frame[TARGET_COL] = model_frame[TARGET_COL].astype(int)
    X_all, y = split_X_y(model_frame)
    groups = model_frame[spec.employee_id_col].astype(str)
    X_all = clean_X_for_model(X_all, spec.employee_id_col)

    return {
        "spec": spec,
        "header_row": header_row,
        "raw_columns": list(raw_df.columns),
        "feature_columns": list(X_all.columns),
        "dropped_columns": dropped_columns,
        "frame": model_frame,
        "X": X_all,
        "y": y,
        "groups": groups,
        "rows": len(model_frame),
        "employees": groups.nunique(),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
    }


def build_time_history_frame(
    raw_df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    drop_exit_metadata: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    data, dropped_columns = build_static_model_frame(
        raw_df,
        spec,
        drop_exit_metadata=drop_exit_metadata,
    )
    if spec.employee_id_col not in data.columns:
        raise ValueError(f"Employee ID column '{spec.employee_id_col}' not found.")

    sort_cols = [spec.employee_id_col]
    if spec.time_col and spec.time_col in data.columns:
        sort_cols.append(spec.time_col)
    elif "year_date" in data.columns:
        sort_cols.append("year_date")
    data = data.sort_values(sort_cols).copy()

    grouped = data.groupby(spec.employee_id_col, sort=False)
    history_features = {
        "history_record_number": grouped.cumcount() + 1,
        "history_prior_records": grouped.cumcount(),
    }

    added_columns = ["history_record_number", "history_prior_records"]
    excluded = {TARGET_COL, spec.employee_id_col}
    numeric_cols = [
        col
        for col in data.select_dtypes(include=[np.number, "bool"]).columns
        if col not in excluded
    ]
    categorical_cols = [
        col
        for col in data.columns
        if col not in excluded and col not in numeric_cols
    ]

    for col in numeric_cols:
        series_group = data.groupby(spec.employee_id_col, sort=False)[col]
        lag_col = f"{col}_hist_prev"
        delta_col = f"{col}_hist_delta_prev"
        mean_col = f"{col}_hist_mean_to_date"
        std_col = f"{col}_hist_std_to_date"
        first_delta_col = f"{col}_hist_delta_first"

        prev = series_group.shift(1)
        history_features[lag_col] = prev
        history_features[delta_col] = data[col] - prev
        history_features[mean_col] = (
            series_group.expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        history_features[std_col] = (
            series_group.expanding()
            .std()
            .reset_index(level=0, drop=True)
        )
        history_features[first_delta_col] = data[col] - series_group.transform("first")
        added_columns.extend([lag_col, delta_col, mean_col, std_col, first_delta_col])

    for col in categorical_cols:
        series_group = data.groupby(spec.employee_id_col, sort=False)[col]
        prev_col = f"{col}_hist_prev"
        changed_col = f"{col}_hist_changed_from_prev"
        nunique_col = f"{col}_hist_unique_to_date"
        prev = series_group.shift(1)
        history_features[prev_col] = prev
        current_text = data[col].astype("string").fillna("__missing__").astype(str)
        prev_text = prev.astype("string").fillna("__missing__").astype(str)
        history_features[changed_col] = ((current_text != prev_text) & prev.notna()).astype(int)
        history_features[nunique_col] = series_group.transform(cumulative_unique_count)
        added_columns.extend([prev_col, changed_col, nunique_col])

    data = pd.concat([data, pd.DataFrame(history_features, index=data.index)], axis=1)
    return data.replace([np.inf, -np.inf], np.nan), dropped_columns, added_columns


def load_time_history_experiment_frame(
    spec: DatasetSpec,
    *,
    drop_exit_metadata: bool = True,
) -> dict[str, Any]:
    raw_df, header_row = read_excel_with_header_detection(spec.path)
    model_frame, dropped_columns, added_columns = build_time_history_frame(
        raw_df,
        spec,
        drop_exit_metadata=drop_exit_metadata,
    )
    model_frame = model_frame.dropna(subset=[TARGET_COL]).copy()
    model_frame[TARGET_COL] = model_frame[TARGET_COL].astype(int)
    X_all, y = split_X_y(model_frame)
    groups = model_frame[spec.employee_id_col].astype(str)
    X_all = clean_X_for_model(X_all, spec.employee_id_col)

    return {
        "spec": spec,
        "header_row": header_row,
        "raw_columns": list(raw_df.columns),
        "feature_columns": list(X_all.columns),
        "dropped_columns": dropped_columns,
        "added_history_columns": added_columns,
        "frame": model_frame,
        "X": X_all,
        "y": y,
        "groups": groups,
        "rows": len(model_frame),
        "employees": groups.nunique(),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
    }


def unavailable_feature_columns(columns) -> list[str]:
    unavailable = []
    bases = [str(col).lower() for col in EXIT_METADATA_COLUMNS]
    for col in columns:
        text = str(col).lower()
        is_unavailable = "aziva" in text
        for base in bases:
            if text == base or text.startswith(f"{base}_hist_"):
                is_unavailable = True
                break
        if is_unavailable:
            unavailable.append(col)
    return unavailable


def mask_unavailable_features(X: pd.DataFrame) -> pd.DataFrame:
    masked = X.copy()
    for col in unavailable_feature_columns(masked.columns):
        if col in masked.columns:
            masked[col] = np.nan
    return masked


def split_counts(y_train, y_val, y_test) -> dict[str, str]:
    def fmt(y):
        return f"{len(y)} rows, {int(y.sum())} positive ({y.mean():.2%})"

    return {
        "Train": fmt(y_train),
        "Val": fmt(y_val),
        "Test": fmt(y_test),
    }


def train_models_on_splits(
    experiment_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    *,
    seed: int,
    builders_fn=model_builders,
    sample_weight_fn=None,
) -> dict[str, Any]:
    rows = []
    best = None
    sample_weight = sample_weight_fn(y_train) if sample_weight_fn else None
    for model_name, estimator_builder in builders_fn(y_train, seed).items():
        pipe = build_pipeline(X_train, estimator_builder())
        if sample_weight is None:
            pipe.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
        train_metrics = evaluate_pipeline(pipe, X_train, y_train)
        val_metrics = evaluate_pipeline(pipe, X_val, y_val)
        test_metrics = evaluate_pipeline(pipe, X_test, y_test)
        row = {
            "Experiment": experiment_name,
            "Model": model_name,
            "Train_AUC": train_metrics["AUC"],
            "Val_AUC": val_metrics["AUC"],
            "Test_AUC": test_metrics["AUC"],
            "Val_F1": val_metrics["F1"],
            "Test_F1": test_metrics["F1"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_Recall@Top10%": test_metrics["Recall@Top10%"],
            "Test_Precision@Top10%": test_metrics["Precision@Top10%"],
            "Test_Recall@Top20%": test_metrics["Recall@Top20%"],
            "Test_Precision@Top20%": test_metrics["Precision@Top20%"],
            "Test_Recall@Top30%": test_metrics["Recall@Top30%"],
            "Test_Precision@Top30%": test_metrics["Precision@Top30%"],
        }
        rows.append(row)
        if best is None or row["Val_AUC"] > best["Val_AUC"]:
            best = {**row, "pipeline": pipe}

    return {
        "name": experiment_name,
        "model_comparison": pd.DataFrame(rows).sort_values("Val_AUC", ascending=False),
        "best_model_name": best["Model"],
        "split_counts": split_counts(y_train, y_val, y_test),
    }


def train_models_with_masked_eval(
    experiment_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    *,
    seed: int,
) -> dict[str, Any]:
    return train_models_on_splits(
        experiment_name,
        X_train,
        mask_unavailable_features(X_val),
        mask_unavailable_features(X_test),
        y_train,
        y_val,
        y_test,
        seed=seed,
    )


def train_single_file(spec: DatasetSpec, *, seed: int) -> dict[str, Any]:
    loaded = load_experiment_frame(spec)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        loaded["X"],
        loaded["y"],
        seed,
        groups=loaded["groups"],
    )
    result = train_models_on_splits(
        spec.tag,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
    )
    return {**result, "data": loaded}


def train_single_file_with_strategy(
    spec: DatasetSpec,
    *,
    seed: int,
    strategy_name: str,
    builders_fn,
    sample_weight_fn=None,
) -> dict[str, Any]:
    loaded = load_experiment_frame(spec)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        loaded["X"],
        loaded["y"],
        seed,
        groups=loaded["groups"],
    )
    result = train_models_on_splits(
        f"{spec.tag}__{strategy_name}",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
        builders_fn=builders_fn,
        sample_weight_fn=sample_weight_fn,
    )
    return {**result, "data": loaded}


def train_single_file_time_history(
    spec: DatasetSpec,
    *,
    seed: int,
    strategy_name: str,
    builders_fn,
) -> dict[str, Any]:
    loaded = load_time_history_experiment_frame(spec)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        loaded["X"],
        loaded["y"],
        seed,
        groups=loaded["groups"],
    )
    result = train_models_on_splits(
        f"{spec.tag}__time_history__{strategy_name}",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
        builders_fn=builders_fn,
    )
    return {**result, "data": loaded}


def train_single_file_train_all_masked_eval(spec: DatasetSpec, *, seed: int) -> dict[str, Any]:
    loaded = load_experiment_frame(spec, drop_exit_metadata=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        loaded["X"],
        loaded["y"],
        seed,
        groups=loaded["groups"],
    )
    result = train_models_with_masked_eval(
        f"{spec.tag}__train_all_masked_eval",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
    )
    result["data"] = {
        **loaded,
        "masked_eval_columns": [col for col in EXIT_METADATA_COLUMNS if col in loaded["X"].columns],
    }
    return result


def train_combined_file1_file2_test_file3(file_data: dict[str, dict[str, Any]], *, seed: int) -> dict[str, Any]:
    train_sources = [file_data["file1"], file_data["file2"]]
    test_source = file_data["file3"]
    feature_columns = sorted(set().union(*(set(src["feature_columns"]) for src in train_sources), set(test_source["feature_columns"])))

    combined_X = pd.concat([src["X"].reindex(columns=feature_columns) for src in train_sources], ignore_index=True)
    combined_y = pd.concat([src["y"] for src in train_sources], ignore_index=True)
    combined_groups = pd.concat(
        [
            pd.Series((f"{src['spec'].tag}:{group}" for group in src["groups"]), dtype="object")
            for src in train_sources
        ],
        ignore_index=True,
    )
    X_train, X_val, y_train, y_val = split_train_val(
        combined_X,
        combined_y,
        seed,
        groups=combined_groups,
    )
    X_test = test_source["X"].reindex(columns=feature_columns)
    y_test = test_source["y"]

    result = train_models_on_splits(
        "file1+file2_train_val__file3_test",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
    )
    result["data"] = {
        "feature_columns": feature_columns,
        "rows": len(combined_X) + len(X_test),
        "train_val_rows": len(combined_X),
        "test_rows": len(X_test),
        "positives": int(combined_y.sum() + y_test.sum()),
    }
    return result


def train_combined_with_strategy(
    file_data: dict[str, dict[str, Any]],
    *,
    seed: int,
    strategy_name: str,
    builders_fn,
    sample_weight_fn=None,
) -> dict[str, Any]:
    train_sources = [file_data["file1"], file_data["file2"]]
    test_source = file_data["file3"]
    feature_columns = sorted(set().union(*(set(src["feature_columns"]) for src in train_sources), set(test_source["feature_columns"])))

    combined_X = pd.concat([src["X"].reindex(columns=feature_columns) for src in train_sources], ignore_index=True)
    combined_y = pd.concat([src["y"] for src in train_sources], ignore_index=True)
    combined_groups = pd.concat(
        [
            pd.Series((f"{src['spec'].tag}:{group}" for group in src["groups"]), dtype="object")
            for src in train_sources
        ],
        ignore_index=True,
    )
    X_train, X_val, y_train, y_val = split_train_val(
        combined_X,
        combined_y,
        seed,
        groups=combined_groups,
    )
    X_test = test_source["X"].reindex(columns=feature_columns)
    y_test = test_source["y"]

    result = train_models_on_splits(
        f"file1+file2_train_val__file3_test__{strategy_name}",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
        builders_fn=builders_fn,
        sample_weight_fn=sample_weight_fn,
    )
    result["data"] = {
        "feature_columns": feature_columns,
        "rows": len(combined_X) + len(X_test),
        "train_val_rows": len(combined_X),
        "test_rows": len(X_test),
        "positives": int(combined_y.sum() + y_test.sum()),
    }
    return result


def train_combined_time_history(
    file_data: dict[str, dict[str, Any]],
    *,
    seed: int,
    strategy_name: str,
    builders_fn,
) -> dict[str, Any]:
    train_sources = [file_data["file1"], file_data["file2"]]
    test_source = file_data["file3"]
    feature_columns = sorted(set().union(*(set(src["feature_columns"]) for src in train_sources), set(test_source["feature_columns"])))

    combined_X = pd.concat([src["X"].reindex(columns=feature_columns) for src in train_sources], ignore_index=True)
    combined_y = pd.concat([src["y"] for src in train_sources], ignore_index=True)
    combined_groups = pd.concat(
        [
            pd.Series((f"{src['spec'].tag}:{group}" for group in src["groups"]), dtype="object")
            for src in train_sources
        ],
        ignore_index=True,
    )
    X_train, X_val, y_train, y_val = split_train_val(
        combined_X,
        combined_y,
        seed,
        groups=combined_groups,
    )
    X_test = test_source["X"].reindex(columns=feature_columns)
    y_test = test_source["y"]

    result = train_models_on_splits(
        f"file1+file2_train_val__file3_test__time_history__{strategy_name}",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
        builders_fn=builders_fn,
    )
    result["data"] = {
        "feature_columns": feature_columns,
        "rows": len(combined_X) + len(X_test),
        "train_val_rows": len(combined_X),
        "test_rows": len(X_test),
        "positives": int(combined_y.sum() + y_test.sum()),
        "added_history_columns": sorted(set().union(*(set(src["added_history_columns"]) for src in train_sources), set(test_source["added_history_columns"]))),
    }
    return result


def train_combined_train_all_masked_eval(file_data: dict[str, dict[str, Any]], *, seed: int) -> dict[str, Any]:
    train_sources = [file_data["file1"], file_data["file2"]]
    test_source = file_data["file3"]
    feature_columns = sorted(set().union(*(set(src["feature_columns"]) for src in train_sources), set(test_source["feature_columns"])))

    combined_X = pd.concat([src["X"].reindex(columns=feature_columns) for src in train_sources], ignore_index=True)
    combined_y = pd.concat([src["y"] for src in train_sources], ignore_index=True)
    combined_groups = pd.concat(
        [
            pd.Series((f"{src['spec'].tag}:{group}" for group in src["groups"]), dtype="object")
            for src in train_sources
        ],
        ignore_index=True,
    )
    X_train, X_val, y_train, y_val = split_train_val(
        combined_X,
        combined_y,
        seed,
        groups=combined_groups,
    )
    X_test = test_source["X"].reindex(columns=feature_columns)
    y_test = test_source["y"]

    result = train_models_with_masked_eval(
        "file1+file2_train_all_masked_eval__file3_test",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=seed,
    )
    result["data"] = {
        "feature_columns": feature_columns,
        "rows": len(combined_X) + len(X_test),
        "train_val_rows": len(combined_X),
        "test_rows": len(X_test),
        "positives": int(combined_y.sum() + y_test.sum()),
        "masked_eval_columns": [col for col in EXIT_METADATA_COLUMNS if col in feature_columns],
    }
    return result


def schema_summary(file_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_columns = {tag: data["raw_columns"] for tag, data in file_data.items()}
    feature_columns = {tag: data["feature_columns"] for tag, data in file_data.items()}
    raw_sets = {tag: set(cols) for tag, cols in raw_columns.items()}
    feature_sets = {tag: set(cols) for tag, cols in feature_columns.items()}
    return {
        "raw_same": len({tuple(cols) for cols in raw_columns.values()}) == 1,
        "feature_same": len({tuple(cols) for cols in feature_columns.values()}) == 1,
        "raw_columns": raw_columns,
        "feature_columns": feature_columns,
        "raw_differences": {
            tag: {
                "missing_vs_file1": sorted(raw_sets["file1"] - cols),
                "extra_vs_file1": sorted(cols - raw_sets["file1"]),
            }
            for tag, cols in raw_sets.items()
        },
        "feature_differences": {
            tag: {
                "missing_vs_file1": sorted(feature_sets["file1"] - cols),
                "extra_vs_file1": sorted(cols - feature_sets["file1"]),
            }
            for tag, cols in feature_sets.items()
        },
    }


def write_experiment_report(
    results: list[dict[str, Any]],
    schema: dict[str, Any],
    output_path: str | Path,
    *,
    seed: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 88,
        "FILEX ROW-LEVEL TURNOVER EXPERIMENTS",
        "=" * 88,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Seed: {seed}",
        "",
        "Preprocessing",
        "- One row remains one observed employee-year row; no latest-row aggregation is applied.",
        "- Model features exclude only employee ID and leaving metadata: aziva_kod, aziva_date, aziva_year, target.",
        "- Remaining raw columns are fed as-is through minimal sklearn handling: constant imputation, numeric scaling, one-hot encoding.",
        "- Splits are by employee ID, so the same worker does not appear in multiple splits.",
        "",
        "Schema check",
        f"- Raw columns identical across all files: {schema['raw_same']}",
        f"- Modeling feature columns identical across all files: {schema['feature_same']}",
    ]

    if not schema["raw_same"]:
        lines.append("- Raw-column differences versus file1:")
        for tag, diff in schema["raw_differences"].items():
            missing = ", ".join(diff["missing_vs_file1"]) or "none"
            extra = ", ".join(diff["extra_vs_file1"]) or "none"
            lines.append(f"  {tag}: missing [{missing}], extra [{extra}]")
    if not schema["feature_same"]:
        lines.append("- Feature-column differences versus file1:")
        for tag, diff in schema["feature_differences"].items():
            missing = ", ".join(diff["missing_vs_file1"]) or "none"
            extra = ", ".join(diff["extra_vs_file1"]) or "none"
            lines.append(f"  {tag}: missing [{missing}], extra [{extra}]")

    for result in results:
        lines.extend(["", "-" * 88, result["name"], "-" * 88])
        data = result.get("data", {})
        if "spec" in data:
            lines.extend(
                [
                    f"Source: {data['spec'].path}",
                    f"Header row used: {data['header_row']}",
                    f"Rows: {data['rows']}",
                    f"Employees: {data['employees']}",
                    f"Positive target rows: {data['positives']} ({data['positive_rate']:.2%})",
                    f"Model features: {len(data['feature_columns'])}",
                    f"Dropped from features: {', '.join(data['dropped_columns']) or 'none'}",
                ]
            )
        else:
            lines.extend(
                [
                    "Train/validation source: file1 + file2",
                    "Test source: file3",
                    f"Train/validation rows before split: {data['train_val_rows']}",
                    f"Test rows: {data['test_rows']}",
                    f"Model features: {len(data['feature_columns'])}",
                ]
            )
        lines.append("Split target counts:")
        for split, text in result["split_counts"].items():
            lines.append(f"  {split}: {text}")
        lines.append("")
        lines.append("Model comparison:")
        lines.append(result["model_comparison"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lines.append(f"Best by validation AUC: {result['best_model_name']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def comparison_rows(baseline_results: list[dict[str, Any]], masked_results: list[dict[str, Any]]) -> pd.DataFrame:
    baseline_map = {
        "file1": "file1__train_all_masked_eval",
        "file2": "file2__train_all_masked_eval",
        "file3": "file3__train_all_masked_eval",
        "file1+file2_train_val__file3_test": "file1+file2_train_all_masked_eval__file3_test",
    }
    masked_by_name = {result["name"]: result for result in masked_results}
    rows = []
    for baseline in baseline_results:
        masked = masked_by_name.get(baseline_map.get(baseline["name"], ""))
        if masked is None:
            continue
        base_best = baseline["model_comparison"].iloc[0]
        masked_best = masked["model_comparison"].iloc[0]
        rows.append(
            {
                "Experiment": baseline["name"],
                "Baseline_Best_Model": baseline["best_model_name"],
                "Baseline_Val_AUC": base_best["Val_AUC"],
                "Baseline_Test_AUC": base_best["Test_AUC"],
                "Baseline_Test_F1": base_best["Test_F1"],
                "Baseline_Test_Precision": base_best["Test_Precision"],
                "Baseline_Test_Recall": base_best["Test_Recall"],
                "Masked_Best_Model": masked["best_model_name"],
                "Masked_Val_AUC": masked_best["Val_AUC"],
                "Masked_Test_AUC": masked_best["Test_AUC"],
                "Masked_Test_F1": masked_best["Test_F1"],
                "Masked_Test_Precision": masked_best["Test_Precision"],
                "Masked_Test_Recall": masked_best["Test_Recall"],
                "Delta_Test_AUC": masked_best["Test_AUC"] - base_best["Test_AUC"],
                "Delta_Test_F1": masked_best["Test_F1"] - base_best["Test_F1"],
                "Delta_Test_Precision": masked_best["Test_Precision"] - base_best["Test_Precision"],
                "Delta_Test_Recall": masked_best["Test_Recall"] - base_best["Test_Recall"],
            }
        )
    return pd.DataFrame(rows)


def best_metric_row(result: dict[str, Any], variant: str, base_experiment: str) -> dict[str, Any]:
    row = result["model_comparison"].iloc[0]
    return {
        "Experiment": base_experiment,
        "Variant": variant,
        "Best_Model": result["best_model_name"],
        "Val_AUC": row["Val_AUC"],
        "Test_AUC": row["Test_AUC"],
        "Test_F1": row["Test_F1"],
        "Test_Precision": row["Test_Precision"],
        "Test_Recall": row["Test_Recall"],
        "Test_Recall@Top10%": row.get("Test_Recall@Top10%", np.nan),
        "Test_Precision@Top10%": row.get("Test_Precision@Top10%", np.nan),
        "Test_Recall@Top20%": row.get("Test_Recall@Top20%", np.nan),
        "Test_Precision@Top20%": row.get("Test_Precision@Top20%", np.nan),
        "Test_Recall@Top30%": row.get("Test_Recall@Top30%", np.nan),
        "Test_Precision@Top30%": row.get("Test_Precision@Top30%", np.nan),
    }


def imbalance_comparison_rows(
    baseline_results: list[dict[str, Any]],
    class_weight_results: list[dict[str, Any]],
    ipw_results: list[dict[str, Any]],
) -> pd.DataFrame:
    baseline_names = ["file1", "file2", "file3", "file1+file2_train_val__file3_test"]
    class_weight_map = {
        "file1": "file1__class_weight",
        "file2": "file2__class_weight",
        "file3": "file3__class_weight",
        "file1+file2_train_val__file3_test": "file1+file2_train_val__file3_test__class_weight",
    }
    ipw_map = {
        "file1": "file1__ipw",
        "file2": "file2__ipw",
        "file3": "file3__ipw",
        "file1+file2_train_val__file3_test": "file1+file2_train_val__file3_test__ipw",
    }
    baseline_by_name = {result["name"]: result for result in baseline_results}
    class_weight_by_name = {result["name"]: result for result in class_weight_results}
    ipw_by_name = {result["name"]: result for result in ipw_results}

    rows = []
    for name in baseline_names:
        if name in baseline_by_name:
            rows.append(best_metric_row(baseline_by_name[name], "baseline", name))
        if class_weight_map[name] in class_weight_by_name:
            rows.append(best_metric_row(class_weight_by_name[class_weight_map[name]], "class_weight", name))
        if ipw_map[name] in ipw_by_name:
            rows.append(best_metric_row(ipw_by_name[ipw_map[name]], "ipw", name))
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    baseline_auc = comparison[comparison["Variant"].eq("baseline")].set_index("Experiment")["Test_AUC"]
    baseline_recall20 = comparison[comparison["Variant"].eq("baseline")].set_index("Experiment")["Test_Recall@Top20%"]
    comparison["Delta_Test_AUC_vs_Baseline"] = comparison.apply(
        lambda row: row["Test_AUC"] - baseline_auc.get(row["Experiment"], np.nan),
        axis=1,
    )
    comparison["Delta_Recall@Top20_vs_Baseline"] = comparison.apply(
        lambda row: row["Test_Recall@Top20%"] - baseline_recall20.get(row["Experiment"], np.nan),
        axis=1,
    )
    return comparison


def time_history_comparison_rows(
    baseline_results: list[dict[str, Any]],
    unweighted_results: list[dict[str, Any]],
    class_weight_results: list[dict[str, Any]],
) -> pd.DataFrame:
    baseline_names = ["file1", "file2", "file3", "file1+file2_train_val__file3_test"]
    unweighted_map = {
        "file1": "file1__time_history__unweighted",
        "file2": "file2__time_history__unweighted",
        "file3": "file3__time_history__unweighted",
        "file1+file2_train_val__file3_test": "file1+file2_train_val__file3_test__time_history__unweighted",
    }
    class_weight_map = {
        "file1": "file1__time_history__class_weight",
        "file2": "file2__time_history__class_weight",
        "file3": "file3__time_history__class_weight",
        "file1+file2_train_val__file3_test": "file1+file2_train_val__file3_test__time_history__class_weight",
    }
    baseline_by_name = {result["name"]: result for result in baseline_results}
    unweighted_by_name = {result["name"]: result for result in unweighted_results}
    class_weight_by_name = {result["name"]: result for result in class_weight_results}

    rows = []
    for name in baseline_names:
        if name in baseline_by_name:
            rows.append(best_metric_row(baseline_by_name[name], "raw_baseline", name))
        if unweighted_map[name] in unweighted_by_name:
            rows.append(best_metric_row(unweighted_by_name[unweighted_map[name]], "time_history_unweighted", name))
        if class_weight_map[name] in class_weight_by_name:
            rows.append(best_metric_row(class_weight_by_name[class_weight_map[name]], "time_history_class_weight", name))

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    baseline_auc = comparison[comparison["Variant"].eq("raw_baseline")].set_index("Experiment")["Test_AUC"]
    baseline_recall20 = comparison[comparison["Variant"].eq("raw_baseline")].set_index("Experiment")["Test_Recall@Top20%"]
    comparison["Delta_Test_AUC_vs_Raw_Baseline"] = comparison.apply(
        lambda row: row["Test_AUC"] - baseline_auc.get(row["Experiment"], np.nan),
        axis=1,
    )
    comparison["Delta_Recall@Top20_vs_Raw_Baseline"] = comparison.apply(
        lambda row: row["Test_Recall@Top20%"] - baseline_recall20.get(row["Experiment"], np.nan),
        axis=1,
    )
    return comparison


def write_train_all_masked_report(
    baseline_results: list[dict[str, Any]],
    masked_results: list[dict[str, Any]],
    schema: dict[str, Any],
    output_path: str | Path,
    *,
    seed: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = comparison_rows(baseline_results, masked_results)

    lines = [
        "=" * 88,
        "TRAIN-WITH-ALL / REAL-LIFE-MASKED EVALUATION",
        "=" * 88,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Seed: {seed}",
        "",
        "Variant definition",
        "- Training features include all raw columns except employee ID and leave_ind.",
        "- Validation/test keep the same feature columns, but real-life unavailable columns are set to missing.",
        f"- Masked validation/test columns: {', '.join(EXIT_METADATA_COLUMNS)}",
        "- This is a stress test. It is not a deployable setup if the model learns to rely on columns that disappear at evaluation time.",
        "",
        "Comparison against baseline real-life model",
        "- Baseline model: exit metadata removed before train/validation/test.",
        "- Masked model: exit metadata present during training, masked during validation/test.",
        "",
        "Schema check",
        f"- Raw columns identical across all files: {schema['raw_same']}",
        f"- Modeling feature columns identical in baseline real-life mode: {schema['feature_same']}",
        "",
        "Best-model comparison",
        "",
    ]
    lines.append(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}") if not comparison.empty else "No comparable rows.")

    for result in masked_results:
        lines.extend(["", "-" * 88, result["name"], "-" * 88])
        data = result.get("data", {})
        if "spec" in data:
            lines.extend(
                [
                    f"Source: {data['spec'].path}",
                    f"Header row used: {data['header_row']}",
                    f"Rows: {data['rows']}",
                    f"Employees: {data['employees']}",
                    f"Positive target rows: {data['positives']} ({data['positive_rate']:.2%})",
                    f"Training features: {len(data['feature_columns'])}",
                    f"Dropped before training: {', '.join(data['dropped_columns']) or 'none'}",
                    f"Masked during validation/test: {', '.join(data['masked_eval_columns']) or 'none'}",
                ]
            )
        else:
            lines.extend(
                [
                    "Train/validation source: file1 + file2",
                    "Test source: file3",
                    f"Train/validation rows before split: {data['train_val_rows']}",
                    f"Test rows: {data['test_rows']}",
                    f"Training features: {len(data['feature_columns'])}",
                    f"Masked during validation/test: {', '.join(data['masked_eval_columns']) or 'none'}",
                ]
            )
        lines.append("Split target counts:")
        for split, text in result["split_counts"].items():
            lines.append(f"  {split}: {text}")
        lines.append("")
        lines.append("Model comparison:")
        lines.append(result["model_comparison"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lines.append(f"Best by validation AUC: {result['best_model_name']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_imbalance_report(
    baseline_results: list[dict[str, Any]],
    class_weight_results: list[dict[str, Any]],
    ipw_results: list[dict[str, Any]],
    output_path: str | Path,
    *,
    seed: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = imbalance_comparison_rows(baseline_results, class_weight_results, ipw_results)

    lines = [
        "=" * 88,
        "CLASS IMBALANCE EXPERIMENTS",
        "=" * 88,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Seed: {seed}",
        "",
        "Variants",
        "- baseline: current real-life feature setup and existing model imbalance handling.",
        "- class_weight: explicit class weights where supported; AdaBoost remains unweighted because class-weighted boosting was unstable on these splits.",
        "- ipw: unweighted estimators trained with stabilized inverse class-prevalence sample weights.",
        "",
        "Top-k metrics",
        "- Recall@TopK%: share of all leavers captured by the highest-risk K% of rows.",
        "- Precision@TopK%: leaver rate inside the highest-risk K% of rows.",
        "- Top-k is threshold-free and is more informative here than the default 0.5 threshold.",
        "",
        "Best-model comparison",
        "",
    ]
    lines.append(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}") if not comparison.empty else "No comparable rows.")

    for title, results in [
        ("CLASS WEIGHT DETAILS", class_weight_results),
        ("IPW DETAILS", ipw_results),
    ]:
        lines.extend(["", "=" * 88, title, "=" * 88])
        for result in results:
            lines.extend(["", "-" * 88, result["name"], "-" * 88])
            data = result.get("data", {})
            if "spec" in data:
                lines.extend(
                    [
                        f"Source: {data['spec'].path}",
                        f"Rows: {data['rows']}",
                        f"Employees: {data['employees']}",
                        f"Positive target rows: {data['positives']} ({data['positive_rate']:.2%})",
                        f"Model features: {len(data['feature_columns'])}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Train/validation source: file1 + file2",
                        "Test source: file3",
                        f"Train/validation rows before split: {data['train_val_rows']}",
                        f"Test rows: {data['test_rows']}",
                        f"Model features: {len(data['feature_columns'])}",
                    ]
                )
            lines.append("Split target counts:")
            for split, text in result["split_counts"].items():
                lines.append(f"  {split}: {text}")
            lines.append("")
            lines.append("Model comparison:")
            lines.append(result["model_comparison"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            lines.append(f"Best by validation AUC: {result['best_model_name']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_time_history_report(
    baseline_results: list[dict[str, Any]],
    unweighted_results: list[dict[str, Any]],
    class_weight_results: list[dict[str, Any]],
    output_path: str | Path,
    *,
    seed: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = time_history_comparison_rows(baseline_results, unweighted_results, class_weight_results)

    lines = [
        "=" * 88,
        "TIME-HISTORY PREPROCESSING EXPERIMENTS",
        "=" * 88,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Seed: {seed}",
        "",
        "Preprocessing",
        "- Keeps row-level employee-year labels, but augments each row with same-employee history features.",
        "- Rows are sorted by employee and calc_month/year_date before feature creation.",
        "- Numeric history features: previous value, delta from previous, expanding mean/std, delta from first observed value.",
        "- Categorical history features: previous value, changed-from-previous flag, cumulative unique value count.",
        "- Exit metadata remains excluded: aziva_kod, aziva_date, aziva_year, target.",
        "",
        "Variants",
        "- raw_baseline: current raw row-level baseline from model_performance.txt.",
        "- time_history_unweighted: history preprocessing with unweighted models.",
        "- time_history_class_weight: history preprocessing with class-weight handling where supported.",
        "",
        "Best-model comparison",
        "",
    ]
    lines.append(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}") if not comparison.empty else "No comparable rows.")

    for title, results in [
        ("TIME HISTORY UNWEIGHTED DETAILS", unweighted_results),
        ("TIME HISTORY CLASS WEIGHT DETAILS", class_weight_results),
    ]:
        lines.extend(["", "=" * 88, title, "=" * 88])
        for result in results:
            lines.extend(["", "-" * 88, result["name"], "-" * 88])
            data = result.get("data", {})
            if "spec" in data:
                lines.extend(
                    [
                        f"Source: {data['spec'].path}",
                        f"Rows: {data['rows']}",
                        f"Employees: {data['employees']}",
                        f"Positive target rows: {data['positives']} ({data['positive_rate']:.2%})",
                        f"Model features: {len(data['feature_columns'])}",
                        f"History features added: {len(data['added_history_columns'])}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Train/validation source: file1 + file2",
                        "Test source: file3",
                        f"Train/validation rows before split: {data['train_val_rows']}",
                        f"Test rows: {data['test_rows']}",
                        f"Model features: {len(data['feature_columns'])}",
                        f"History features added: {len(data['added_history_columns'])}",
                    ]
                )
            lines.append("Split target counts:")
            for split, text in result["split_counts"].items():
                lines.append(f"  {split}: {text}")
            lines.append("")
            lines.append("Model comparison:")
            lines.append(result["model_comparison"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            lines.append(f"Best by validation AUC: {result['best_model_name']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_filex_experiments(*, seed: int = 42, output_dir: str | Path = "output") -> dict[str, Any]:
    specs = [spec_for_path(path, path.stem) for path in FILEX_PATHS]
    file_data = {spec.tag: load_experiment_frame(spec) for spec in specs}
    individual_results = [train_single_file(spec, seed=seed) for spec in specs]
    combined_result = train_combined_file1_file2_test_file3(file_data, seed=seed)
    results = individual_results + [combined_result]
    schema = schema_summary(file_data)
    output_path = write_experiment_report(
        results,
        schema,
        Path(output_dir) / "model_performance.txt",
        seed=seed,
    )
    return {
        "results": results,
        "schema": schema,
        "output_path": output_path,
    }


def run_train_all_masked_experiments(
    baseline_result: dict[str, Any] | None = None,
    *,
    seed: int = 42,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    specs = [spec_for_path(path, path.stem) for path in FILEX_PATHS]
    if baseline_result is None:
        baseline_result = run_filex_experiments(seed=seed, output_dir=output_dir)

    all_feature_data = {
        spec.tag: load_experiment_frame(spec, drop_exit_metadata=False)
        for spec in specs
    }
    individual_results = [
        train_single_file_train_all_masked_eval(spec, seed=seed)
        for spec in specs
    ]
    combined_result = train_combined_train_all_masked_eval(all_feature_data, seed=seed)
    results = individual_results + [combined_result]
    output_path = write_train_all_masked_report(
        baseline_result["results"],
        results,
        baseline_result["schema"],
        Path(output_dir) / "model_performance_train_all_masked.txt",
        seed=seed,
    )
    return {
        "results": results,
        "schema": baseline_result["schema"],
        "output_path": output_path,
        "comparison": comparison_rows(baseline_result["results"], results),
    }


def run_imbalance_experiments(
    baseline_result: dict[str, Any] | None = None,
    *,
    seed: int = 42,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    specs = [spec_for_path(path, path.stem) for path in FILEX_PATHS]
    if baseline_result is None:
        baseline_result = run_filex_experiments(seed=seed, output_dir=output_dir)

    file_data = {spec.tag: load_experiment_frame(spec) for spec in specs}
    class_weight_results = [
        train_single_file_with_strategy(
            spec,
            seed=seed,
            strategy_name="class_weight",
            builders_fn=class_weight_model_builders,
        )
        for spec in specs
    ]
    class_weight_results.append(
        train_combined_with_strategy(
            file_data,
            seed=seed,
            strategy_name="class_weight",
            builders_fn=class_weight_model_builders,
        )
    )

    ipw_results = [
        train_single_file_with_strategy(
            spec,
            seed=seed,
            strategy_name="ipw",
            builders_fn=ipw_model_builders,
            sample_weight_fn=inverse_probability_weights,
        )
        for spec in specs
    ]
    ipw_results.append(
        train_combined_with_strategy(
            file_data,
            seed=seed,
            strategy_name="ipw",
            builders_fn=ipw_model_builders,
            sample_weight_fn=inverse_probability_weights,
        )
    )

    output_path = write_imbalance_report(
        baseline_result["results"],
        class_weight_results,
        ipw_results,
        Path(output_dir) / "model_performance_imbalance.txt",
        seed=seed,
    )
    comparison = imbalance_comparison_rows(
        baseline_result["results"],
        class_weight_results,
        ipw_results,
    )
    return {
        "class_weight_results": class_weight_results,
        "ipw_results": ipw_results,
        "output_path": output_path,
        "comparison": comparison,
    }


def run_time_history_experiments(
    baseline_result: dict[str, Any] | None = None,
    *,
    seed: int = 42,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    specs = [spec_for_path(path, path.stem) for path in FILEX_PATHS]
    if baseline_result is None:
        baseline_result = run_filex_experiments(seed=seed, output_dir=output_dir)

    time_file_data = {spec.tag: load_time_history_experiment_frame(spec) for spec in specs}
    unweighted_results = [
        train_single_file_time_history(
            spec,
            seed=seed,
            strategy_name="unweighted",
            builders_fn=ipw_model_builders,
        )
        for spec in specs
    ]
    unweighted_results.append(
        train_combined_time_history(
            time_file_data,
            seed=seed,
            strategy_name="unweighted",
            builders_fn=ipw_model_builders,
        )
    )

    class_weight_results = [
        train_single_file_time_history(
            spec,
            seed=seed,
            strategy_name="class_weight",
            builders_fn=class_weight_model_builders,
        )
        for spec in specs
    ]
    class_weight_results.append(
        train_combined_time_history(
            time_file_data,
            seed=seed,
            strategy_name="class_weight",
            builders_fn=class_weight_model_builders,
        )
    )

    output_path = write_time_history_report(
        baseline_result["results"],
        unweighted_results,
        class_weight_results,
        Path(output_dir) / "model_performance_time_history.txt",
        seed=seed,
    )
    comparison = time_history_comparison_rows(
        baseline_result["results"],
        unweighted_results,
        class_weight_results,
    )
    return {
        "unweighted_results": unweighted_results,
        "class_weight_results": class_weight_results,
        "output_path": output_path,
        "comparison": comparison,
    }


def train_static_clean(
    spec: DatasetSpec,
    *,
    feature_set: str = "all_static",
    seed: int = 42,
) -> dict[str, Any]:
    del feature_set
    return train_single_file(spec, seed=seed)
