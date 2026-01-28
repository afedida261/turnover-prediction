import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .base_model import BaseTurnoverModel


class LogisticRegressionTurnover(BaseTurnoverModel):
    """Logistic Regression model - good for interpretability."""
    def __init__(self, class_weight='balanced', max_iter=1000, random_state=42, **kwargs):
        super().__init__()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight=class_weight, max_iter=max_iter, random_state=random_state, **kwargs)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self):
        """Return coefficients as feature importance (absolute value for magnitude)."""
        if self.feature_names is not None:
            # Return actual coefficients (not absolute) to see direction
            return dict(zip(self.feature_names, self.model.coef_[0]))
        return self.model.coef_[0]


class RandomForestTurnover(BaseTurnoverModel):
    def __init__(self, n_estimators=100, class_weight='balanced', random_state=42, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state, **kwargs)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        if self.feature_names is not None:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return self.model.feature_importances_

class XGBoostTurnover(BaseTurnoverModel):
    def __init__(self, scale_pos_weight=None, use_label_encoder=False, eval_metric='logloss', random_state=42, **kwargs):
        super().__init__()
        self.model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=use_label_encoder, eval_metric=eval_metric, random_state=random_state, **kwargs)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        try:
            importance_dict = self.model.get_booster().get_score(importance_type='gain')
            # Normalize the importance scores
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v / total for k, v in importance_dict.items()}
            return importance_dict
        except:
             if self.feature_names is not None:
                 importances = self.model.feature_importances_
                 # Normalize
                 total = sum(importances)
                 if total > 0:
                     importances = importances / total
                 return dict(zip(self.feature_names, importances))
             return self.model.feature_importances_

class AdaBoostTurnover(BaseTurnoverModel):
    """AdaBoost classifier - good for reducing bias and handling imbalanced data."""
    def __init__(self, n_estimators=200, learning_rate=0.5, random_state=42, **kwargs):
        super().__init__()
        # Use a deeper decision tree as base estimator for better performance
        base_estimator = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=random_state)
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """Return feature importance from AdaBoost."""
        if self.feature_names is not None:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return self.model.feature_importances_


class EnsembleTurnover(BaseTurnoverModel):
    def __init__(self, models=None, random_state=42):
        super().__init__()
        if models is None:
            self.estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10, random_state=random_state)),
                ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=random_state))
            ]
        else:
            self.estimators = models
            
        self.model = VotingClassifier(estimators=self.estimators, voting='soft')
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """Return averaged feature importance from ensemble members."""
        if self.feature_names is None:
            return {"info": "Ensemble model - refer to individual base learners"}
        
        # Average feature importance from RF and AdaBoost (XGBoost uses different format)
        importance = np.zeros(len(self.feature_names))
        count = 0
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importance += estimator.feature_importances_
                count += 1
        
        if count > 0:
            importance /= count
            return dict(zip(self.feature_names, importance))
        return {"info": "Ensemble model - refer to individual base learners"}
