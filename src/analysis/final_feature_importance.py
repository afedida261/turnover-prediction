"""Explain the locked final model and generate final feature-importance outputs.

Methods:
- Native XGBoost gain importance.
- TreeSHAP on a reproducible file3 sample.
- Grouped transformed-feature permutation on file3, shuffled within period.
- Employee-grouped drop-column CV AUC gain on file1 + file2 only.

File3 is never used to select the model or tune its threshold. Its permutation
results describe external reliance only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imputations import PAYMENT_COLUMNS
from src.preprocess import prepare_turnover_data
from src.final_modeling import (
    CandidateSpec,
    candidate_feature_columns,
    make_candidate_pipeline,
    positive_probabilities,
)


sns.set_theme(style="whitegrid")


def transformed_feature_names(pipeline) -> list[str]:
    return pipeline.named_steps["tabular"].get_feature_names_out().tolist()


def transform_for_model(pipeline, X: pd.DataFrame):
    values = X
    if "payment_imputer" in pipeline.named_steps:
        values = pipeline.named_steps["payment_imputer"].transform(values)
    return pipeline.named_steps["tabular"].transform(values)


def dense_array(values) -> np.ndarray:
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def map_transformed_to_source(name: str, source_columns: list[str]) -> str:
    normalized = str(name)
    if normalized.startswith("missingindicator_"):
        normalized = normalized[len("missingindicator_"):]
    if normalized in source_columns:
        return normalized
    for source in sorted(source_columns, key=len, reverse=True):
        if normalized.startswith(f"{source}_"):
            return source
    return normalized


def classify_domain(feature: str) -> tuple[str, str]:
    text = str(feature).lower()
    if "was_missing" in text or "missing_raw" in text or "observed_fraction" in text:
        return "Data availability", "Descriptive"
    if text.startswith("history_") or "elapsed_month" in text or "months_since" in text:
        return "Data maturity / timing", "Descriptive"
    if any(token in text for token in ("payment", "salary", "sahar")):
        return "Compensation", "Actionable"
    if any(token in text for token in ("illness", "hedrut")):
        return "Absence / wellness", "Partially actionable"
    if any(token in text for token in ("omes", "workload", "workhours")):
        return "Workload", "Actionable"
    if any(token in text for token in ("manager", "maneger", "tafkid", "mahala", "seif", "maamad", "contract")):
        return "Role / manager / organization", "Partially actionable"
    if any(token in text for token in ("age", "vetek", "tenure", "career_start")):
        return "Age / tenure", "Descriptive"
    if any(token in text for token in ("yishuv", "distance", "commute")):
        return "Location / commute", "Descriptive"
    if any(token in text for token in ("children", "gender", "marital")):
        return "Demographic context", "Descriptive"
    return "Other", "Review"


def add_classification(frame: pd.DataFrame, feature_column: str = "Source_Feature") -> pd.DataFrame:
    data = frame.copy()
    classifications = data[feature_column].map(classify_domain)
    data["Domain"] = classifications.map(lambda item: item[0])
    data["Actionability"] = classifications.map(lambda item: item[1])
    data["History_Feature"] = (
        data[feature_column].astype(str).str.contains("_hist_")
        | data[feature_column].astype(str).str.startswith("history_")
    )
    return data


def aggregate_importance(
    transformed: pd.DataFrame,
    *,
    value_column: str,
    source_columns: list[str],
) -> pd.DataFrame:
    data = transformed.copy()
    data["Source_Feature"] = data["Transformed_Feature"].map(
        lambda value: map_transformed_to_source(value, source_columns)
    )
    grouped = (
        data.groupby("Source_Feature", as_index=False)
        .agg(
            Importance_Sum=(value_column, "sum"),
            Importance_Max=(value_column, "max"),
            Transformed_Feature_Count=("Transformed_Feature", "size"),
        )
        .sort_values("Importance_Sum", ascending=False)
        .reset_index(drop=True)
    )
    grouped.insert(0, "Rank", np.arange(1, len(grouped) + 1))
    return add_classification(grouped)


def native_importance(pipeline, source_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = transformed_feature_names(pipeline)
    values = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    transformed = pd.DataFrame({
        "Transformed_Feature": names,
        "Native_Importance": values,
    }).sort_values("Native_Importance", ascending=False)
    transformed.insert(0, "Rank", np.arange(1, len(transformed) + 1))
    source = aggregate_importance(
        transformed,
        value_column="Native_Importance",
        source_columns=source_columns,
    ).rename(columns={
        "Importance_Sum": "Native_Importance_Sum",
        "Importance_Max": "Native_Importance_Max",
    })
    return transformed, source


def shap_importance(
    pipeline,
    X: pd.DataFrame,
    source_columns: list[str],
    *,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    import xgboost as xgb

    sample = X.sample(min(sample_size, len(X)), random_state=seed)
    transformed_values = dense_array(transform_for_model(pipeline, sample))
    names = transformed_feature_names(pipeline)
    transformed_frame = pd.DataFrame(transformed_values, columns=names, index=sample.index)
    contributions = pipeline.named_steps["model"].get_booster().predict(
        xgb.DMatrix(transformed_values), pred_contribs=True
    )[:, :-1]
    mean_abs = np.abs(contributions).mean(axis=0)
    transformed = pd.DataFrame({
        "Transformed_Feature": names,
        "Mean_Abs_SHAP": mean_abs,
        "Mean_SHAP": contributions.mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    transformed.insert(0, "Rank", np.arange(1, len(transformed) + 1))

    direction_rows = []
    for index, name in enumerate(names):
        values = transformed_values[:, index]
        shap_values = contributions[:, index]
        finite = np.isfinite(values) & np.isfinite(shap_values)
        correlation = (
            np.corrcoef(values[finite], shap_values[finite])[0, 1]
            if finite.sum() > 2
            and np.std(values[finite]) > 0
            and np.std(shap_values[finite]) > 0
            else np.nan
        )
        direction_rows.append((name, correlation))
    direction = dict(direction_rows)
    transformed["Value_SHAP_Correlation"] = transformed["Transformed_Feature"].map(direction)

    source = aggregate_importance(
        transformed,
        value_column="Mean_Abs_SHAP",
        source_columns=source_columns,
    ).rename(columns={
        "Importance_Sum": "Mean_Abs_SHAP_Sum",
        "Importance_Max": "Mean_Abs_SHAP_Max",
    })
    return transformed, source, transformed_frame, contributions


def bootstrap_ci(values, *, seed: int, repeats: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def within_period_permutation_indices(periods: pd.Series, rng: np.random.Generator) -> np.ndarray:
    period_values = pd.to_datetime(periods, errors="coerce").astype("string").fillna("Missing")
    result = np.arange(len(period_values))
    for _, positions in pd.Series(np.arange(len(period_values))).groupby(period_values.to_numpy()):
        indices = positions.to_numpy()
        result[indices] = rng.permutation(indices)
    return result


def transformed_group_permutation(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    periods: pd.Series,
    source_features: list[str],
    source_columns: list[str],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    transformed = dense_array(transform_for_model(pipeline, X))
    names = transformed_feature_names(pipeline)
    mapped = [map_transformed_to_source(name, source_columns) for name in names]
    model = pipeline.named_steps["model"]
    base_probabilities = model.predict_proba(transformed)[:, 1]
    base_auc = roc_auc_score(y, base_probabilities)
    rng = np.random.default_rng(seed)
    rows = []

    for feature in source_features:
        column_indices = [idx for idx, source in enumerate(mapped) if source == feature]
        if not column_indices:
            continue
        drops = []
        for _ in range(repeats):
            permutation = within_period_permutation_indices(periods.reset_index(drop=True), rng)
            changed = transformed.copy()
            changed[:, column_indices] = transformed[permutation][:, column_indices]
            permuted_auc = roc_auc_score(y, model.predict_proba(changed)[:, 1])
            drops.append(base_auc - permuted_auc)
        low, high = bootstrap_ci(drops, seed=seed)
        rows.append({
            "Source_Feature": feature,
            "Transformed_Feature_Count": len(column_indices),
            "Base_File3_AUC": base_auc,
            "Permutation_AUC_Drop_Mean": float(np.mean(drops)),
            "Permutation_AUC_Drop_CI_Low": low,
            "Permutation_AUC_Drop_CI_High": high,
            "Positive_Drop_Fraction": float(np.mean(np.asarray(drops) > 0)),
            "Repeats": repeats,
        })
    result = pd.DataFrame(rows).sort_values("Permutation_AUC_Drop_Mean", ascending=False)
    return add_classification(result)


def source_stratified_group_folds(X, y, groups, *, folds: int, seed: int):
    summary = pd.DataFrame({
        "group": groups.astype(str).to_numpy(),
        "target": y.astype(int).to_numpy(),
        "source": X["source"].astype(str).to_numpy(),
    }).groupby("group", sort=False).agg(target=("target", "max"), source=("source", "first"))
    labels = summary["source"] + "_" + summary["target"].astype(str)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    group_names = summary.index.to_numpy()
    result = []
    for train_group_idx, validation_group_idx in splitter.split(group_names, labels):
        train_groups = set(group_names[train_group_idx])
        validation_groups = set(group_names[validation_group_idx])
        train_mask = groups.astype(str).isin(train_groups).to_numpy()
        validation_mask = groups.astype(str).isin(validation_groups).to_numpy()
        result.append((train_mask, validation_mask))
    return result


def resolve_drop_feature(feature: str, X: pd.DataFrame) -> str | None:
    if feature in X.columns:
        return feature
    if feature.endswith("_was_missing"):
        base = feature[: -len("_was_missing")]
        if base in X.columns:
            return base
    return None


def drop_column_cv_gain(
    prepared,
    artifact,
    features: list[str],
    *,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    spec = CandidateSpec(artifact["model_name"], artifact["payment_strategy"])
    splits = source_stratified_group_folds(
        prepared.X_train,
        prepared.y_train,
        prepared.train_groups,
        folds=folds,
        seed=seed,
    )
    baseline = []
    for fold, (train_mask, validation_mask) in enumerate(splits, start=1):
        X_train = prepared.X_train.loc[train_mask]
        y_train = prepared.y_train.loc[train_mask]
        X_validation = prepared.X_train.loc[validation_mask]
        y_validation = prepared.y_train.loc[validation_mask]
        columns = candidate_feature_columns(X_train, spec)
        pipeline = make_candidate_pipeline(spec, X_train[columns], y_train, seed + fold)
        pipeline.fit(X_train[columns], y_train)
        auc = roc_auc_score(y_validation, positive_probabilities(pipeline, X_validation[columns]))
        baseline.append((fold, train_mask, validation_mask, auc))

    rows = []
    for requested_feature in features:
        feature = resolve_drop_feature(requested_feature, prepared.X_train)
        if feature is None:
            continue
        deltas = []
        without_scores = []
        full_scores = []
        for fold, train_mask, validation_mask, full_auc in baseline:
            X_train = prepared.X_train.loc[train_mask]
            y_train = prepared.y_train.loc[train_mask]
            X_validation = prepared.X_train.loc[validation_mask]
            y_validation = prepared.y_train.loc[validation_mask]
            columns = [
                column for column in candidate_feature_columns(X_train, spec)
                if column != feature
            ]
            pipeline = make_candidate_pipeline(spec, X_train[columns], y_train, seed + fold)
            pipeline.fit(X_train[columns], y_train)
            without_auc = roc_auc_score(
                y_validation,
                positive_probabilities(pipeline, X_validation[columns]),
            )
            full_scores.append(full_auc)
            without_scores.append(without_auc)
            deltas.append(full_auc - without_auc)
        low, high = bootstrap_ci(deltas, seed=seed)
        rows.append({
            "Requested_Source_Feature": requested_feature,
            "Dropped_Raw_Feature": feature,
            "CV_Folds": folds,
            "Full_CV_AUC_Mean": float(np.mean(full_scores)),
            "Without_Feature_CV_AUC_Mean": float(np.mean(without_scores)),
            "AUC_Gain_Mean": float(np.mean(deltas)),
            "AUC_Gain_CI_Low": low,
            "AUC_Gain_CI_High": high,
            "Fold_AUC_Gains": "; ".join(f"{value:.5f}" for value in deltas),
        })
    result = pd.DataFrame(rows).sort_values("AUC_Gain_Mean", ascending=False)
    return add_classification(result, "Requested_Source_Feature")


def consensus_importance(
    native: pd.DataFrame,
    shap_source: pd.DataFrame,
    permutation: pd.DataFrame,
    drop_column: pd.DataFrame,
) -> pd.DataFrame:
    summary = native[["Source_Feature", "Native_Importance_Sum"]].merge(
        shap_source[["Source_Feature", "Mean_Abs_SHAP_Sum"]], on="Source_Feature", how="outer"
    )
    summary = summary.merge(
        permutation[["Source_Feature", "Permutation_AUC_Drop_Mean"]],
        on="Source_Feature", how="left",
    )
    if not drop_column.empty:
        drop_values = drop_column[["Requested_Source_Feature", "AUC_Gain_Mean"]].rename(
            columns={"Requested_Source_Feature": "Source_Feature"}
        )
        summary = summary.merge(drop_values, on="Source_Feature", how="left")
    else:
        summary["AUC_Gain_Mean"] = np.nan

    metrics = [
        "Native_Importance_Sum",
        "Mean_Abs_SHAP_Sum",
        "Permutation_AUC_Drop_Mean",
        "AUC_Gain_Mean",
    ]
    rank_columns = []
    for metric in metrics:
        rank = f"{metric}_Percentile"
        positive = summary[metric].clip(lower=0)
        summary[rank] = positive.rank(pct=True, ascending=True)
        rank_columns.append(rank)
    summary["Consensus_Score"] = summary[rank_columns].mean(axis=1, skipna=True)
    summary = summary.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    summary.insert(0, "Consensus_Rank", np.arange(1, len(summary) + 1))
    return add_classification(summary)

def save_grouped_consensus_plot(consensus: pd.DataFrame, path: Path) -> pd.DataFrame:
    poster_labels = {
        "Age / tenure": "Tenure & age",
        "Role / manager / organization": "Role, manager & org",
        "Location / commute": "Location & commute",
        "Data maturity / timing": "Data timing",
        "Data availability": "Data availability",
        "Demographic context": "Demographic context",
        "Absence / wellness": "Absence & wellness",
    }
    data = consensus.copy()
    data["Feature_Group"] = data["Domain"].map(lambda value: poster_labels.get(value, value))
    data.loc[data["Feature_Group"].isin(["Other", "Demographic context"]), "Feature_Group"] = "Other context"

    metrics = [
        "Native_Importance_Sum",
        "Mean_Abs_SHAP_Sum",
        "Permutation_AUC_Drop_Mean",
        "AUC_Gain_Mean",
    ]
    clipped = data[["Feature_Group", *metrics]].copy()
    for metric in metrics:
        clipped[metric] = clipped[metric].clip(lower=0)

    grouped = clipped.groupby("Feature_Group", as_index=False).agg(
        **{metric: (metric, lambda values: values.sum(min_count=1)) for metric in metrics},
        Feature_Count=("Feature_Group", "size"),
    )

    rank_columns = []
    for metric in metrics:
        rank = f"{metric}_Percentile"
        grouped[rank] = grouped[metric].rank(pct=True, ascending=True)
        rank_columns.append(rank)
    grouped["Consensus_Score"] = grouped[rank_columns].mean(axis=1, skipna=True)
    grouped = grouped.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    grouped.insert(0, "Consensus_Rank", np.arange(1, len(grouped) + 1))

    plot = grouped.sort_values("Consensus_Score", ascending=True)
    colors = [
        "#2f6f9f",
        "#2a9d8f",
        "#e9a03f",
        "#c95f5f",
        "#6f63b6",
        "#6a8f3a",
        "#8c6d5b",
        "#5f7a8a",
    ][: len(plot)]
    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.58 * len(plot) + 1.6)))
    bars = ax.barh(plot["Feature_Group"], plot["Consensus_Score"], color=colors, alpha=0.96)
    ax.set_xlim(0, 1)
    ax.set_title("Consensus importance by feature group", fontsize=18, pad=12)
    ax.set_xlabel("Consensus score across importance methods", fontsize=13)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for bar, value in zip(bars, plot["Consensus_Score"]):
        ax.text(
            min(value + 0.015, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value < 0.93 else "right",
            fontsize=12,
            color="#222222",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return grouped

def save_horizontal_bar(
    data: pd.DataFrame,
    feature: str,
    value: str,
    path: Path,
    title: str,
    xlabel: str,
    *,
    top_n: int = 20,
    low: str | None = None,
    high: str | None = None,
) -> Path:
    plot = data.dropna(subset=[value]).sort_values(value, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(plot) + 1.6)))
    errors = None
    if low and high:
        center = plot[value].to_numpy(float)
        errors = np.vstack([
            np.maximum(center - plot[low].to_numpy(float), 0),
            np.maximum(plot[high].to_numpy(float) - center, 0),
        ])
    ax.barh(plot[feature].astype(str), plot[value], xerr=errors, color="#356aa0", alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_shap_beeswarm(X: pd.DataFrame, values: np.ndarray, path: Path, *, top_n: int) -> Path:
    mean_abs = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]
    rng = np.random.default_rng(42)
    figure, axis = plt.subplots(figsize=(11, max(5, 0.38 * len(top_indices) + 1.8)))
    labels = []
    scatter = None
    for position, index in enumerate(top_indices[::-1]):
        feature_values = X.iloc[:, index].to_numpy(dtype=float)
        contributions = values[:, index]
        jitter = rng.normal(0, 0.075, size=len(contributions))
        finite_values = feature_values[np.isfinite(feature_values)]
        if len(finite_values):
            low, high = np.percentile(finite_values, [5, 95])
            colors = np.clip(feature_values, low, high)
        else:
            colors = np.zeros_like(feature_values)
        scatter = axis.scatter(
            contributions,
            np.full(len(contributions), position) + jitter,
            c=colors,
            cmap="coolwarm",
            s=11,
            alpha=0.60,
            edgecolors="none",
        )
        labels.append(str(X.columns[index]))
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xlabel("SHAP contribution to model score")
    axis.set_title("SHAP beeswarm summary")
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
        colorbar.set_label("Transformed feature value")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def save_dependence_grid(
    X: pd.DataFrame,
    values: np.ndarray,
    transformed_importance: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 6,
) -> Path:
    features = transformed_importance.head(top_n)["Transformed_Feature"].tolist()
    indices = {feature: idx for idx, feature in enumerate(X.columns)}
    rows = int(np.ceil(len(features) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        idx = indices[feature]
        ax.scatter(X[feature], values[:, idx], s=12, alpha=0.45, color="#356aa0")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("Transformed feature value")
        ax.set_ylabel("SHAP value")
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_domain_plot(shap_source: pd.DataFrame, path: Path) -> pd.DataFrame:
    domain = (
        shap_source.groupby(["Domain", "Actionability"], as_index=False)["Mean_Abs_SHAP_Sum"]
        .sum()
        .sort_values("Mean_Abs_SHAP_Sum", ascending=False)
    )
    total = domain["Mean_Abs_SHAP_Sum"].sum()
    domain["SHAP_Share"] = domain["Mean_Abs_SHAP_Sum"] / total if total else 0
    save_horizontal_bar(
        domain,
        "Domain",
        "SHAP_Share",
        path,
        "Importance by feature domain",
        "Share of mean absolute SHAP",
        top_n=len(domain),
    )
    return domain


def save_actionability_plot(domain: pd.DataFrame, path: Path) -> pd.DataFrame:
    actionability = (
        domain.groupby("Actionability", as_index=False)["Mean_Abs_SHAP_Sum"]
        .sum()
        .sort_values("Mean_Abs_SHAP_Sum", ascending=False)
    )
    total = actionability["Mean_Abs_SHAP_Sum"].sum()
    actionability["SHAP_Share"] = actionability["Mean_Abs_SHAP_Sum"] / total if total else 0
    save_horizontal_bar(
        actionability,
        "Actionability",
        "SHAP_Share",
        path,
        "Importance by actionability",
        "Share of mean absolute SHAP",
        top_n=len(actionability),
    )
    return actionability


def save_performance_plot(comparison: pd.DataFrame, path: Path) -> Path:
    plot = comparison[["Candidate", "Val_PR_AUC", "Test_PR_AUC", "Val_AUC", "Test_AUC"]].copy()
    plot = plot.sort_values("Val_PR_AUC", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot.set_index("Candidate")[["Val_PR_AUC", "Test_PR_AUC"]].plot.barh(ax=axes[0])
    axes[0].set_title("PR-AUC: validation vs file3")
    axes[0].set_xlabel("PR-AUC")
    plot.set_index("Candidate")[["Val_AUC", "Test_AUC"]].plot.barh(ax=axes[1])
    axes[1].set_title("ROC-AUC: validation vs file3")
    axes[1].set_xlabel("ROC-AUC")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def write_report(
    path: Path,
    *,
    comparison: pd.DataFrame,
    artifact: dict,
    prepared,
    native_source: pd.DataFrame,
    shap_source: pd.DataFrame,
    permutation: pd.DataFrame,
    drop_column: pd.DataFrame,
    consensus: pd.DataFrame,
    domain: pd.DataFrame,
    actionability: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    best = comparison.loc[comparison["Selected_Best"].astype(bool)].iloc[0]
    file3_rate = float(prepared.y_test.mean())
    pr_lift = float(best["Test_PR_AUC"] / file3_rate)
    top20_lift = float(best["Test_Precision@Top20%"] / file3_rate)
    rf_test_pr = comparison.loc[
        comparison["Candidate"].eq("Random Forest__learned_imputation"), "Test_PR_AUC"
    ].iloc[0]
    lines = [
        "FINAL MODEL REVIEW AND FEATURE IMPORTANCE",
        "=" * 96,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Locked model: {artifact['candidate']}",
        "",
        "Results review",
        f"- Validation selected the model by PR-AUC ({best['Val_PR_AUC']:.4f}); validation ROC-AUC was {best['Val_AUC']:.4f}.",
        f"- On untouched file3: ROC-AUC={best['Test_AUC']:.4f}, PR-AUC={best['Test_PR_AUC']:.4f}, F1={best['Test_F1']:.4f}.",
        f"- File3 prevalence is {file3_rate:.2%}; PR-AUC is {pr_lift:.2f}x prevalence.",
        f"- Top 20% review captures {best['Test_Recall@Top20%']:.2%} of positive periods at {best['Test_Precision@Top20%']:.2%} precision ({top20_lift:.2f}x baseline precision).",
        f"- Random Forest with imputation has higher file3 PR-AUC ({rf_test_pr:.4f}) but was not validation-selected; changing winners now would be test-set selection.",
        "",
        "Interpretation rules",
        "- Native gain shows how XGBoost used splits; it can favor high-cardinality/redundant variables.",
        "- Mean absolute SHAP shows contribution magnitude, not causality.",
        "- File3 permutation measures external reliance by shuffling transformed feature groups within period.",
        "- Drop-column CV retrains without a feature on file1/file2; positive AUC gain means the feature helped on average.",
        "- Correlated history features can share or substitute importance, so conclusions should be made at both feature and domain level.",
        "",
        "Top native source features",
        native_source.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Top SHAP source features",
        shap_source.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "External file3 permutation reliance",
        permutation.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Employee-grouped drop-column CV AUC gain",
        drop_column.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Consensus ranking",
        consensus.head(25).to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Domain summary",
        domain.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Actionability summary",
        actionability.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Graphs",
    ]
    lines.extend(f"- {plot}" for plot in plot_paths)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_final_feature_importance(
    *,
    output_dir: str | Path = "output/final/feature_importance",
    artifact_path: str | Path = "artifacts/final_best_model.pkl",
    top_n: int = 20,
    effect_top_n: int = 6,
    cv_folds: int = 3,
    permutation_repeats: int = 20,
    shap_sample_size: int = 800,
    seed: int = 42,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(artifact_path)
    if artifact["model_name"] != "XGBoost":
        raise ValueError("This final analysis expects the locked XGBoost winner.")

    prepared = prepare_turnover_data()
    columns = artifact["feature_columns"]
    pipeline = artifact["pipeline"]
    source_columns = columns + [
        f"{column}_was_missing" for column in PAYMENT_COLUMNS if column in columns
    ]

    native_transformed, native_source = native_importance(pipeline, source_columns)
    shap_transformed, shap_source, shap_X, shap_values = shap_importance(
        pipeline,
        prepared.X_test[columns],
        source_columns,
        sample_size=shap_sample_size,
        seed=seed,
    )

    ranking = shap_source.merge(
        native_source[["Source_Feature", "Native_Importance_Sum"]],
        on="Source_Feature",
        how="left",
    )
    ranking["SHAP_Rank"] = ranking["Mean_Abs_SHAP_Sum"].rank(ascending=False)
    ranking["Native_Rank"] = ranking["Native_Importance_Sum"].rank(ascending=False)
    ranking["Initial_Mean_Rank"] = ranking[["SHAP_Rank", "Native_Rank"]].mean(axis=1)
    top_features = ranking.sort_values("Initial_Mean_Rank")["Source_Feature"].head(top_n).tolist()

    permutation = transformed_group_permutation(
        pipeline,
        prepared.X_test[columns],
        prepared.y_test.reset_index(drop=True),
        prepared.test_frame["calc_month"].reset_index(drop=True),
        top_features,
        source_columns,
        repeats=permutation_repeats,
        seed=seed,
    )
    drop_candidates = []
    for feature in top_features:
        resolved = resolve_drop_feature(feature, prepared.X_train)
        if resolved and feature not in drop_candidates:
            drop_candidates.append(feature)
        if len(drop_candidates) >= effect_top_n:
            break
    drop_column = drop_column_cv_gain(
        prepared,
        artifact,
        drop_candidates,
        folds=cv_folds,
        seed=seed,
    )
    consensus = consensus_importance(native_source, shap_source, permutation, drop_column)

    native_transformed.to_csv(output / "native_importance_transformed.csv", index=False)
    native_source.to_csv(output / "native_importance_source.csv", index=False)
    shap_transformed.to_csv(output / "shap_importance_transformed.csv", index=False)
    shap_source.to_csv(output / "shap_importance_source.csv", index=False)
    permutation.to_csv(output / "permutation_reliance_file3.csv", index=False)
    drop_column.to_csv(output / "drop_column_cv_auc_gain.csv", index=False)
    consensus.to_csv(output / "consensus_importance.csv", index=False)

    plot_paths = [
        save_horizontal_bar(
            native_source, "Source_Feature", "Native_Importance_Sum",
            output / "native_top_features.png", "Native XGBoost feature importance", "Gain importance", top_n=top_n,
        ),
        save_horizontal_bar(
            shap_source, "Source_Feature", "Mean_Abs_SHAP_Sum",
            output / "shap_top_features.png", "Mean absolute SHAP by source feature", "Mean |SHAP|", top_n=top_n,
        ),
        save_shap_beeswarm(shap_X, shap_values, output / "shap_beeswarm.png", top_n=top_n),
        save_dependence_grid(
            shap_X, shap_values, shap_transformed, output / "shap_dependence_top6.png", top_n=6
        ),
        save_horizontal_bar(
            permutation,
            "Source_Feature",
            "Permutation_AUC_Drop_Mean",
            output / "permutation_reliance_file3.png",
            "File3 permutation reliance (within-period shuffle)",
            "ROC-AUC drop",
            top_n=top_n,
            low="Permutation_AUC_Drop_CI_Low",
            high="Permutation_AUC_Drop_CI_High",
        ),
        save_horizontal_bar(
            drop_column,
            "Requested_Source_Feature",
            "AUC_Gain_Mean",
            output / "drop_column_cv_auc_gain.png",
            "Employee-grouped drop-column CV gain",
            "AUC gain from retaining feature",
            top_n=effect_top_n,
            low="AUC_Gain_CI_Low",
            high="AUC_Gain_CI_High",
        ),
        save_horizontal_bar(
            consensus,
            "Source_Feature",
            "Consensus_Score",
            output / "consensus_top_features.png",
            "Consensus feature ranking",
            "Mean percentile rank across available methods",
            top_n=top_n,
        ),
    ]

    grouped_consensus = save_grouped_consensus_plot(
        consensus, output / "consensus_grouped_features.png"
    )

    domain = save_domain_plot(shap_source, output / "importance_by_domain.png")
    actionability = save_actionability_plot(domain, output / "importance_by_actionability.png")
    grouped_consensus.to_csv(output / "consensus_grouped_features.csv", index=False)
    domain.to_csv(output / "domain_importance.csv", index=False)
    actionability.to_csv(output / "actionability_importance.csv", index=False)
    plot_paths.extend([
        output / "consensus_grouped_features.png",
        output / "importance_by_domain.png",
        output / "importance_by_actionability.png",
    ])

    comparison = pd.read_csv("output/final/model_comparison.csv")
    plot_paths.append(save_performance_plot(comparison, output / "model_performance_comparison.png"))
    write_report(
        output / "feature_importance_report.txt",
        comparison=comparison,
        artifact=artifact,
        prepared=prepared,
        native_source=native_source,
        shap_source=shap_source,
        permutation=permutation,
        drop_column=drop_column,
        consensus=consensus,
        domain=domain,
        actionability=actionability,
        plot_paths=plot_paths,
    )
    metadata = {
        "artifact": str(artifact_path),
        "candidate": artifact["candidate"],
        "top_n": top_n,
        "drop_column_features": drop_candidates,
        "cv_folds": cv_folds,
        "permutation_repeats": permutation_repeats,
        "shap_sample_size": min(shap_sample_size, len(prepared.X_test)),
        "cleaning_audit": asdict(prepared.cleaning_audit),
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "output_dir": output,
        "native_source": native_source,
        "shap_source": shap_source,
        "permutation": permutation,
        "drop_column": drop_column,
        "consensus": consensus,
        "domain": domain,
        "report": output / "feature_importance_report.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain the locked final turnover model.")
    parser.add_argument("--output-dir", default="output/final/feature_importance")
    parser.add_argument("--artifact", default="artifacts/final_best_model.pkl")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--effect-top-n", type=int, default=6)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--shap-sample-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_final_feature_importance(
        output_dir=args.output_dir,
        artifact_path=args.artifact,
        top_n=args.top_n,
        effect_top_n=args.effect_top_n,
        cv_folds=args.cv_folds,
        permutation_repeats=args.permutation_repeats,
        shap_sample_size=args.shap_sample_size,
        seed=args.seed,
    )
    print(f"Report: {result['report']}")
    print("Top consensus features:")
    print(
        result["consensus"].head(15)[
            ["Consensus_Rank", "Source_Feature", "Domain", "Consensus_Score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()


