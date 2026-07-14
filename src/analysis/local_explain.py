"""
local_explain.py
----------------
Per-employee (local) and population (global) explanations of turnover-risk
predictions for the dashboard chatbot.

The primary path uses SHAP on the fitted estimator inside the inference
pipeline, aggregating one-hot / transformed columns back to the original
human-readable features. If SHAP is unavailable or fails for a given model,
a robust model-agnostic sensitivity fallback is used (neutralise one feature
at a time to the population typical value and measure the change in predicted
risk).

All public helpers return plain strings ready to hand to the chatbot LLM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_DESCRIPTIONS

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - shap optional
    SHAP_AVAILABLE = False


# Original features that make sense to explain / neutralise for HR.
NUMERIC_FEATURES = ["avg_Payment", "avg_omes", "avg_illness", "vetek_months", "age"]
CATEGORICAL_FEATURES = [
    "Maamad", "Seif", "contract_type", "TeurGroupHscm", "gender", "EMP_Matzav_Mishpachti",
]

# Simple per-session caches (module persists for the Streamlit process).
_GLOBAL_CACHE: dict[int, list[tuple[str, float]]] = {}


def _readable(col: str) -> str:
    return FEATURE_DESCRIPTIONS.get(col, col)


def _fmt_value(col: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if col in ("avg_omes",):
        try:
            return f"{float(value) * 100:.0f}% of full-time"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    return str(value)


def _split_pipeline(pipeline):
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return None, pipeline
    if len(steps) == 1:
        return None, steps[-1][1]
    from sklearn.pipeline import Pipeline
    return Pipeline(steps[:-1]), steps[-1][1]


def _densify(array) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float64)


def _transformed_names(pre, n_features: int) -> list[str]:
    if pre is not None:
        try:
            return list(pre.get_feature_names_out())
        except Exception:
            pass
    return [f"f{i}" for i in range(n_features)]


def _map_to_original(name: str, feature_columns: list[str]) -> str:
    """Map a transformed / one-hot column name back to its source feature."""
    best = None
    for col in feature_columns:
        if name == col or name.startswith(f"{col}_"):
            if best is None or len(col) > len(best):
                best = col
    return best or name


def _raw_shap_values(estimator, X_transformed: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Return SHAP values (n_samples, n_features) for the positive (leave) class."""
    # XGBoost — use the native booster (avoids SHAP/XGBoost version mismatches).
    try:
        from xgboost import XGBClassifier
        if isinstance(estimator, XGBClassifier):
            import xgboost as xgb
            contribs = estimator.get_booster().predict(xgb.DMatrix(X_transformed), pred_contribs=True)
            return np.asarray(contribs)[:, :-1]  # drop bias column
    except Exception:
        pass

    if not SHAP_AVAILABLE:
        raise RuntimeError("shap is not installed")

    import shap
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression

    if isinstance(estimator, (RandomForestClassifier, AdaBoostClassifier)):
        values = shap.TreeExplainer(estimator).shap_values(X_transformed)
        values = np.array(values)
        if isinstance(values, list):
            values = np.array(values[1])
        if values.ndim == 3:
            values = values[:, :, 1]
        return values

    if isinstance(estimator, LogisticRegression):
        return np.asarray(shap.LinearExplainer(estimator, background).shap_values(X_transformed))

    # Generic (e.g. VotingClassifier) — permutation explainer on churn probability.
    explainer = shap.PermutationExplainer(lambda data: estimator.predict_proba(data)[:, 1], background)
    return np.asarray(explainer(X_transformed, max_evals=400).values)


def _aggregate(values_row: np.ndarray, names: list[str], feature_columns: list[str]) -> dict[str, float]:
    agg: dict[str, float] = {}
    for name, val in zip(names, values_row):
        original = _map_to_original(name, feature_columns)
        agg[original] = agg.get(original, 0.0) + float(val)
    return agg


def _sensitivity_contributions(api, records: pd.DataFrame, background_df: pd.DataFrame) -> dict[str, float]:
    """Model-agnostic fallback: neutralise one feature at a time and measure Δrisk."""
    base_prob, _ = api.predict_risk(records)
    latest = records.copy()
    time_col = getattr(api, "time_col", None)
    if time_col and time_col in latest.columns:
        latest = latest.sort_values(time_col)
    idx = latest.index[-1]

    agg: dict[str, float] = {}
    candidates = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c in records.columns]
    for col in candidates:
        neutral: Any = None
        if col in NUMERIC_FEATURES and col in background_df.columns:
            series = pd.to_numeric(background_df[col], errors="coerce")
            if series.notna().any():
                neutral = float(series.median())
        elif col in background_df.columns:
            mode = background_df[col].mode()
            neutral = mode.iloc[0] if len(mode) else None
        if neutral is None:
            continue
        modified = latest.copy()
        modified.at[idx, col] = neutral
        try:
            p, _ = api.predict_risk(modified)
            agg[col] = base_prob - p  # positive => current value raises risk vs typical
        except Exception:
            continue
    return agg


def _typical_value(background_df: pd.DataFrame, col: str):
    if col not in background_df.columns:
        return None
    if col in NUMERIC_FEATURES:
        series = pd.to_numeric(background_df[col], errors="coerce")
        return float(series.median()) if series.notna().any() else None
    mode = background_df[col].mode()
    return mode.iloc[0] if len(mode) else None


def explain_employee_prediction(
    api,
    employee_records: pd.DataFrame,
    background_df: pd.DataFrame,
    employee_id: str = "",
    top_n: int = 8,
) -> str:
    """Explain why a single employee's predicted risk is high or low."""
    if employee_records is None or employee_records.empty:
        return "No records available for that employee, so I can't explain the prediction."

    feature_columns = list(getattr(api, "feature_names", []))
    time_col = getattr(api, "time_col", None)

    try:
        base_prob, base_cat = api.predict_risk(employee_records)
    except Exception as exc:
        return f"Could not score that employee: {exc}"

    latest = employee_records.copy()
    if time_col and time_col in latest.columns:
        latest = latest.sort_values(time_col)
    latest_row = latest.tail(1)

    method = "SHAP feature contributions"
    contributions: dict[str, float] = {}
    try:
        if not feature_columns:
            raise RuntimeError("model feature columns unknown")
        pipeline = api.model
        pre, estimator = _split_pipeline(pipeline)
        X = latest_row.reindex(columns=feature_columns)
        bg = background_df.reindex(columns=feature_columns)
        bg_sample = bg.sample(min(100, len(bg)), random_state=42) if len(bg) else bg
        if pre is not None:
            X_t = _densify(pre.transform(X))
            bg_t = _densify(pre.transform(bg_sample)) if len(bg_sample) else X_t
        else:
            X_t = _densify(X.values)
            bg_t = _densify(bg_sample.values) if len(bg_sample) else X_t
        names = _transformed_names(pre, X_t.shape[1])
        shap_row = np.asarray(_raw_shap_values(estimator, X_t, bg_t))[0]
        contributions = _aggregate(shap_row, names, feature_columns)
    except Exception:
        contributions = _sensitivity_contributions(api, employee_records, background_df)
        method = "risk sensitivity (neutralise-one-feature)"

    # Keep only meaningful, explainable original features.
    explainable = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    contributions = {k: v for k, v in contributions.items() if k in explainable and abs(v) > 1e-9}
    if not contributions:
        return (
            f"Employee {employee_id or ''}: predicted risk {base_prob*100:.1f}% ({base_cat}). "
            "No single feature stands out as a strong driver — the score reflects a mix of small effects."
        )

    total = sum(abs(v) for v in contributions.values()) or 1.0
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    latest_series = latest_row.iloc[0]

    up_lines, down_lines = [], []
    for col, val in ranked:
        share = abs(val) / total * 100
        strength = "strong" if share >= 25 else ("moderate" if share >= 10 else "minor")
        emp_val = _fmt_value(col, latest_series.get(col))
        typ_val = _fmt_value(col, _typical_value(background_df, col))
        line = f"  - {_readable(col)}: {emp_val} (typical {typ_val}) | {strength} ({share:.0f}%)"
        if val > 0:
            up_lines.append(line)
        else:
            down_lines.append(line)

    parts = [f"Employee {employee_id or ''}: predicted risk {base_prob*100:.1f}% ({base_cat})."]
    if up_lines:
        parts.append("Factors pushing risk UP:\n" + "\n".join(up_lines))
    if down_lines:
        parts.append("Factors pulling risk DOWN:\n" + "\n".join(down_lines))
    parts.append(f"(Explanation method: {method}.)")
    return "\n".join(parts)


def global_feature_importance(api, background_df: pd.DataFrame, top_n: int = 15, sample: int = 150) -> str:
    """Rank the features that drive turnover risk across the whole population (SHAP)."""
    feature_columns = list(getattr(api, "feature_names", []))
    if not feature_columns:
        return "Model feature information is unavailable, so I can't rank global drivers."

    pipeline = api.model
    cache_key = id(pipeline)
    if cache_key in _GLOBAL_CACHE:
        ranked = _GLOBAL_CACHE[cache_key]
    else:
        try:
            pre, estimator = _split_pipeline(pipeline)
            src = background_df.reindex(columns=feature_columns)
            samp = src.sample(min(sample, len(src)), random_state=42) if len(src) else src
            if samp.empty:
                return "No data available to compute global feature importance."
            X_t = _densify(pre.transform(samp)) if pre is not None else _densify(samp.values)
            names = _transformed_names(pre, X_t.shape[1])
            shap_values = np.asarray(_raw_shap_values(estimator, X_t, X_t))
            mean_abs = np.abs(shap_values).mean(axis=0)
            agg = _aggregate(mean_abs, names, feature_columns)
            ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
            _GLOBAL_CACHE[cache_key] = ranked
        except Exception as exc:
            return f"Could not compute SHAP global importance: {exc}"

    top = ranked[:top_n]
    if not top:
        return "No feature importance could be computed."
    total = sum(v for _, v in ranked) or 1.0
    lines = [
        f"  {i}. {_readable(col)}: {v / total * 100:.1f}% of total impact"
        for i, (col, v) in enumerate(top, 1)
    ]
    return "Top turnover-risk drivers across the organisation (SHAP mean |impact|):\n" + "\n".join(lines)
