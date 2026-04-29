"""
shap_explainability.py
-----------------------
SHAP-based feature importance for turnover prediction models.
Supports TreeExplainer (RF, XGBoost, AdaBoost) and LinearExplainer (LogisticRegression).
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config import FEATURE_DESCRIPTIONS


def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Returns an appropriate SHAP explainer for the given BaseTurnoverModel wrapper.

    Args:
        model: A BaseTurnoverModel instance (has .model attribute with raw estimator).
        X_background: DataFrame used as background sample for kernel/permutation explainers.

    Returns:
        (explainer, use_legacy_api) tuple.
        use_legacy_api=True  → call explainer.shap_values(X)
        use_legacy_api=False → call explainer(X) and read .values
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    raw = model.model

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression

    try:
        from xgboost import XGBClassifier
        xgb_types = (XGBClassifier,)
    except ImportError:
        xgb_types = ()

    # VotingClassifier — permutation explainer on predict_proba (churn column)
    if isinstance(raw, VotingClassifier):
        background = X_background.sample(min(50, len(X_background)), random_state=42)
        def proba_churn(X):
            return raw.predict_proba(X)[:, 1]
        return shap.PermutationExplainer(proba_churn, background), False

    # XGBoost — use native booster pred_contribs (avoids SHAP/XGBoost version mismatch)
    if isinstance(raw, xgb_types):
        return None, 'xgb_native'

    # Random Forest / AdaBoost — TreeExplainer
    if isinstance(raw, (RandomForestClassifier, AdaBoostClassifier)):
        return shap.TreeExplainer(raw), True

    # Logistic Regression — LinearExplainer
    if isinstance(raw, LogisticRegression):
        background = X_background.sample(min(100, len(X_background)), random_state=42)
        return shap.LinearExplainer(raw, background), True

    raise TypeError(f"No SHAP explainer defined for model type: {type(raw)}")


def compute_shap_summary(model, X: pd.DataFrame, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """
    Computes mean |SHAP| per feature and returns a ranked summary DataFrame.

    Args:
        model: A BaseTurnoverModel instance.
        X: Feature DataFrame (aligned to feature_names, already preprocessed).
        feature_names: List of feature column names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns: Rank, Feature, Readable_Name, Mean_Abs_SHAP
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    # Use a background sample for efficiency; force all columns to float64
    sample_size = min(150, len(X))
    X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
    X_sample = X_sample.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)

    explainer, use_legacy_api = get_shap_explainer(model, X_sample)

    if use_legacy_api == 'xgb_native':
        # Use XGBoost's built-in SHAP (bypasses shap.TreeExplainer version issues)
        import xgboost as xgb
        dmat = xgb.DMatrix(X_sample)
        shap_values = model.model.get_booster().predict(dmat, pred_contribs=True)
        shap_values = shap_values[:, :-1]  # drop bias (last) column
    elif use_legacy_api:
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    else:
        # PermutationExplainer returns shap.Explanation object
        raw_out = explainer(X_sample, max_evals=500)
        shap_values = raw_out.values
        
    # Enforce 2D array: (samples, features)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        # Some SHAP versions return (samples, features, classes) for TreeExplainer
        shap_values = shap_values[:, :, 1]
        
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    results = pd.DataFrame({
        'Feature': feature_names,
        'Readable_Name': [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names],
        'Mean_Abs_SHAP': mean_abs_shap,
    })

    results = results.sort_values('Mean_Abs_SHAP', ascending=False).head(top_n).reset_index(drop=True)
    results.insert(0, 'Rank', results.index + 1)

    return results
