"""Actionable feature importance within age and tenure groups.

This analysis answers a narrower question than the global feature-importance
report: after descriptive age/tenure features are known to matter, which
management-relevant features still carry signal inside comparable age or tenure
segments?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_feature_importance import (
    add_classification,
    classify_domain,
    dense_array,
    map_transformed_to_source,
    transform_for_model,
    transformed_feature_names,
    within_period_permutation_indices,
)
from src.imputations import PAYMENT_COLUMNS
from src.preprocess import prepare_turnover_data


sns.set_theme(style="whitegrid")


ACTIONABLE_LABELS = {"Actionable", "Partially actionable"}


def age_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[17, 34, 49, 59, np.inf],
        labels=["18-34", "35-49", "50-59", "60+"],
        right=True,
    ).astype("string").fillna("Unknown")


def tenure_band(values: pd.Series) -> pd.Series:
    years = pd.to_numeric(values, errors="coerce") / 12
    return pd.cut(
        years,
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=["<2y", "2-5y", "5-10y", "10y+"],
        right=False,
    ).astype("string").fillna("Unknown")


def transformed_contributions(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    import xgboost as xgb

    transformed = dense_array(transform_for_model(pipeline, X))
    names = transformed_feature_names(pipeline)
    contributions = pipeline.named_steps["model"].get_booster().predict(
        xgb.DMatrix(transformed), pred_contribs=True
    )[:, :-1]
    return pd.DataFrame(transformed, columns=names, index=X.index), contributions


def shap_by_group(
    *,
    groups: pd.Series,
    names: list[str],
    contributions: np.ndarray,
    source_columns: list[str],
    y: pd.Series,
    top_n: int,
) -> pd.DataFrame:
    mapped = pd.Series([map_transformed_to_source(name, source_columns) for name in names], index=names)
    rows = []
    for group_name, positions in groups.reset_index(drop=True).groupby(groups.reset_index(drop=True)).groups.items():
        indices = np.asarray(list(positions), dtype=int)
        group_y = y.iloc[indices]
        if not len(indices):
            continue
        transformed_importance = pd.DataFrame({
            "Transformed_Feature": names,
            "Source_Feature": mapped.to_numpy(),
            "Mean_Abs_SHAP": np.abs(contributions[indices, :]).mean(axis=0),
        })
        source = (
            transformed_importance.groupby("Source_Feature", as_index=False)
            .agg(
                Mean_Abs_SHAP_Sum=("Mean_Abs_SHAP", "sum"),
                Transformed_Feature_Count=("Transformed_Feature", "size"),
            )
            .sort_values("Mean_Abs_SHAP_Sum", ascending=False)
        )
        source = add_classification(source)
        source = source[source["Actionability"].isin(ACTIONABLE_LABELS)].head(top_n).copy()
        source.insert(0, "Rank_In_Group", np.arange(1, len(source) + 1))
        source.insert(0, "Prevalence", float(group_y.mean()))
        source.insert(0, "Positive_Rows", int(group_y.sum()))
        source.insert(0, "Rows", int(len(group_y)))
        source.insert(0, "Group", str(group_name))
        rows.append(source)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def actionable_candidates(shap_grouped: pd.DataFrame, top_n: int) -> list[str]:
    if shap_grouped.empty:
        return []
    return (
        shap_grouped.groupby("Source_Feature", as_index=False)["Mean_Abs_SHAP_Sum"]
        .mean()
        .sort_values("Mean_Abs_SHAP_Sum", ascending=False)
        ["Source_Feature"]
        .head(top_n)
        .tolist()
    )


def permutation_by_group(
    *,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    periods: pd.Series,
    groups: pd.Series,
    source_columns: list[str],
    features: list[str],
    repeats: int,
    seed: int,
    min_positive: int,
    min_negative: int,
) -> pd.DataFrame:
    transformed = dense_array(transform_for_model(pipeline, X))
    names = transformed_feature_names(pipeline)
    mapped = [map_transformed_to_source(name, source_columns) for name in names]
    model = pipeline.named_steps["model"]
    base_probabilities = model.predict_proba(transformed)[:, 1]
    rng = np.random.default_rng(seed)
    rows = []

    group_series = groups.reset_index(drop=True)
    for group_name, positions in group_series.groupby(group_series).groups.items():
        group_indices = np.asarray(list(positions), dtype=int)
        group_y = y.iloc[group_indices].reset_index(drop=True)
        positives = int(group_y.sum())
        negatives = int(len(group_y) - positives)
        prevalence = float(group_y.mean()) if len(group_y) else np.nan
        can_score = positives >= min_positive and negatives >= min_negative
        base_pr = average_precision_score(group_y, base_probabilities[group_indices]) if can_score else np.nan
        base_auc = roc_auc_score(group_y, base_probabilities[group_indices]) if can_score else np.nan

        for feature in features:
            column_indices = [idx for idx, source in enumerate(mapped) if source == feature]
            if not column_indices:
                continue
            pr_drops = []
            auc_drops = []
            if can_score:
                group_periods = periods.iloc[group_indices].reset_index(drop=True)
                group_transformed = transformed[group_indices, :]
                for _ in range(repeats):
                    permutation = within_period_permutation_indices(group_periods, rng)
                    changed = group_transformed.copy()
                    changed[:, column_indices] = group_transformed[permutation][:, column_indices]
                    probabilities = model.predict_proba(changed)[:, 1]
                    pr_drops.append(base_pr - average_precision_score(group_y, probabilities))
                    auc_drops.append(base_auc - roc_auc_score(group_y, probabilities))
            domain, actionability = classify_domain(feature)
            rows.append({
                "Group": str(group_name),
                "Rows": int(len(group_y)),
                "Positive_Rows": positives,
                "Prevalence": prevalence,
                "Source_Feature": feature,
                "Domain": domain,
                "Actionability": actionability,
                "Base_PR_AUC": base_pr,
                "Base_ROC_AUC": base_auc,
                "Permutation_PR_AUC_Drop_Mean": float(np.mean(pr_drops)) if pr_drops else np.nan,
                "Permutation_ROC_AUC_Drop_Mean": float(np.mean(auc_drops)) if auc_drops else np.nan,
                "Positive_PR_Drop_Fraction": float(np.mean(np.asarray(pr_drops) > 0)) if pr_drops else np.nan,
                "Repeats": repeats if can_score else 0,
                "Skipped_Reason": "" if can_score else f"needs >= {min_positive} positives and >= {min_negative} negatives",
            })
    return pd.DataFrame(rows)


def save_grouped_bar(data: pd.DataFrame, path: Path, value: str, title: str, top_n: int) -> Path:
    plot_data = data.copy()
    if plot_data.empty:
        return path
    plot_data = plot_data.groupby("Group", group_keys=False).head(top_n)
    height = max(5, 0.32 * len(plot_data) + 1.5)
    plt.figure(figsize=(14, height))
    sns.barplot(data=plot_data, x=value, y="Source_Feature", hue="Group", dodge=False)
    plt.title(title)
    plt.xlabel(value.replace("_", " "))
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_report(
    path: Path,
    *,
    age_shap: pd.DataFrame,
    tenure_shap: pd.DataFrame,
    age_perm: pd.DataFrame,
    tenure_perm: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    lines = [
        "GROUPED ACTIONABLE FEATURE IMPORTANCE",
        "=" * 88,
        "",
        "What this does",
        "- Slices untouched file3 into age bands and tenure bands.",
        "- Ranks only actionable and partially actionable features inside each slice.",
        "- SHAP ranking is descriptive model contribution inside the slice, not causality.",
        "- Permutation PR-AUC/ROC-AUC drop shows whether the slice's predictions rely on that feature.",
        "- Slices with too few positives are skipped for permutation because PR-AUC becomes unstable.",
        "",
        "Top actionable SHAP features by age band",
        age_shap.head(80).to_string(index=False, float_format=lambda value: f"{value:.5f}") if not age_shap.empty else "No rows.",
        "",
        "Top actionable SHAP features by tenure band",
        tenure_shap.head(80).to_string(index=False, float_format=lambda value: f"{value:.5f}") if not tenure_shap.empty else "No rows.",
        "",
        "Actionable permutation reliance by age band",
        age_perm.to_string(index=False, float_format=lambda value: f"{value:.5f}") if not age_perm.empty else "No rows.",
        "",
        "Actionable permutation reliance by tenure band",
        tenure_perm.to_string(index=False, float_format=lambda value: f"{value:.5f}") if not tenure_perm.empty else "No rows.",
        "",
        "Graphs",
    ]
    lines.extend(f"- {plot}" for plot in plot_paths)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_grouped_actionable_importance(
    *,
    output_dir: str | Path = "output/final/feature_importance/grouped_actionable",
    artifact_path: str | Path = "artifacts/final_best_model.pkl",
    top_n: int = 8,
    permutation_top_n: int = 10,
    permutation_repeats: int = 20,
    min_positive: int = 8,
    min_negative: int = 40,
    seed: int = 42,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(artifact_path)
    prepared = prepare_turnover_data()
    columns = artifact["feature_columns"]
    source_columns = columns + [f"{column}_was_missing" for column in PAYMENT_COLUMNS if column in columns]
    pipeline = artifact["pipeline"]

    X = prepared.X_test[columns].copy()
    y = prepared.y_test.reset_index(drop=True)
    frame = prepared.test_frame.reset_index(drop=True)
    transformed, contributions = transformed_contributions(pipeline, X)
    names = transformed.columns.tolist()
    ages = age_band(frame["age"])
    tenures = tenure_band(frame["vetek_months"])

    age_shap = shap_by_group(
        groups=ages, names=names, contributions=contributions, source_columns=source_columns, y=y, top_n=top_n
    )
    tenure_shap = shap_by_group(
        groups=tenures, names=names, contributions=contributions, source_columns=source_columns, y=y, top_n=top_n
    )
    candidates = sorted(set(actionable_candidates(age_shap, permutation_top_n)) | set(actionable_candidates(tenure_shap, permutation_top_n)))

    age_perm = permutation_by_group(
        pipeline=pipeline,
        X=X,
        y=y,
        periods=frame["calc_month"],
        groups=ages,
        source_columns=source_columns,
        features=candidates,
        repeats=permutation_repeats,
        seed=seed,
        min_positive=min_positive,
        min_negative=min_negative,
    ).sort_values(["Group", "Permutation_PR_AUC_Drop_Mean"], ascending=[True, False])
    tenure_perm = permutation_by_group(
        pipeline=pipeline,
        X=X,
        y=y,
        periods=frame["calc_month"],
        groups=tenures,
        source_columns=source_columns,
        features=candidates,
        repeats=permutation_repeats,
        seed=seed + 1,
        min_positive=min_positive,
        min_negative=min_negative,
    ).sort_values(["Group", "Permutation_PR_AUC_Drop_Mean"], ascending=[True, False])

    age_shap.to_csv(output / "actionable_shap_by_age_band.csv", index=False)
    tenure_shap.to_csv(output / "actionable_shap_by_tenure_band.csv", index=False)
    age_perm.to_csv(output / "actionable_permutation_by_age_band.csv", index=False)
    tenure_perm.to_csv(output / "actionable_permutation_by_tenure_band.csv", index=False)

    plot_paths = [
        save_grouped_bar(
            age_shap,
            output / "actionable_shap_by_age_band.png",
            "Mean_Abs_SHAP_Sum",
            "Top actionable SHAP features within age bands",
            top_n=top_n,
        ),
        save_grouped_bar(
            tenure_shap,
            output / "actionable_shap_by_tenure_band.png",
            "Mean_Abs_SHAP_Sum",
            "Top actionable SHAP features within tenure bands",
            top_n=top_n,
        ),
        save_grouped_bar(
            age_perm.dropna(subset=["Permutation_PR_AUC_Drop_Mean"]),
            output / "actionable_permutation_by_age_band.png",
            "Permutation_PR_AUC_Drop_Mean",
            "Actionable permutation PR-AUC drop within age bands",
            top_n=top_n,
        ),
        save_grouped_bar(
            tenure_perm.dropna(subset=["Permutation_PR_AUC_Drop_Mean"]),
            output / "actionable_permutation_by_tenure_band.png",
            "Permutation_PR_AUC_Drop_Mean",
            "Actionable permutation PR-AUC drop within tenure bands",
            top_n=top_n,
        ),
    ]
    write_report(
        output / "grouped_actionable_importance_report.txt",
        age_shap=age_shap,
        tenure_shap=tenure_shap,
        age_perm=age_perm,
        tenure_perm=tenure_perm,
        plot_paths=plot_paths,
    )
    metadata = {
        "artifact": str(artifact_path),
        "candidate": artifact["candidate"],
        "top_n": top_n,
        "permutation_top_n": permutation_top_n,
        "permutation_repeats": permutation_repeats,
        "min_positive": min_positive,
        "min_negative": min_negative,
        "actionable_permutation_features": candidates,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "output_dir": output,
        "age_shap": age_shap,
        "tenure_shap": tenure_shap,
        "age_permutation": age_perm,
        "tenure_permutation": tenure_perm,
        "report": output / "grouped_actionable_importance_report.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain actionable features within age/tenure groups.")
    parser.add_argument("--output-dir", default="output/final/feature_importance/grouped_actionable")
    parser.add_argument("--artifact", default="artifacts/final_best_model.pkl")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--permutation-top-n", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--min-positive", type=int, default=8)
    parser.add_argument("--min-negative", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_grouped_actionable_importance(
        output_dir=args.output_dir,
        artifact_path=args.artifact,
        top_n=args.top_n,
        permutation_top_n=args.permutation_top_n,
        permutation_repeats=args.permutation_repeats,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        seed=args.seed,
    )
    print(f"Report: {result['report']}")
    print("Top age-band actionable SHAP rows:")
    print(result["age_shap"].head(20).to_string(index=False))
    print("Top tenure-band actionable SHAP rows:")
    print(result["tenure_shap"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()


