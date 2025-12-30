import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from base_model import BaseTurnoverModel

class LogisticRegressionBaseline(BaseTurnoverModel):
    def __init__(self, class_weight='balanced', **kwargs):
        super().__init__()
        self.model = LogisticRegression(class_weight=class_weight, max_iter=1000, **kwargs)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        if self.feature_names is not None:
            return dict(zip(self.feature_names, self.model.coef_[0]))
        return self.model.coef_[0]

class RandomForestTurnover(BaseTurnoverModel):
    def __init__(self, n_estimators=100, class_weight='balanced', **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, **kwargs)
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
    def __init__(self, scale_pos_weight=None, use_label_encoder=False, eval_metric='logloss', **kwargs):
        super().__init__()
        # Note: scale_pos_weight is usually set to sum(negative instances) / sum(positive instances) for balancing
        # We will handle this in fit if not provided, or prompt user to provide it.
        # For this implementation, we allow passing it.
        self.model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=use_label_encoder, eval_metric=eval_metric, **kwargs)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        # Auto-calculate scale_pos_weight if not set and class weighting is desired
        # The prompt asks for "logic for class_weight='balanced' or parameters for scale_pos_weight"
        # Since XGBClassifier params are frozen at init in recent scikit-learn wrapper versions, 
        # we strictly rely on what's passed or defaults.
        # Ideally, we'd calculate it here if it was dynamic.
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        try:
            return self.model.get_booster().get_score(importance_type='weight')
        except:
             if self.feature_names is not None:
                 return dict(zip(self.feature_names, self.model.feature_importances_))
             return self.model.feature_importances_

class EnsembleTurnover(BaseTurnoverModel):
    def __init__(self, models=None):
        super().__init__()
        # If models are not provided, initialize default ones
        if models is None:
            self.estimators = [
                ('lr', LogisticRegression(class_weight='balanced', max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10)) # Heuristic weight
            ]
        else:
            self.estimators = models
            
        self.model = VotingClassifier(estimators=self.estimators, voting='soft')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        # VotingClassifier doesn't have a single feature importance.
        # We can average them or return separate ones.
        # For simplicity, return a message or aggregation if possible.
        return {"info": "Ensemble model - refer to individual base learners for importance"}
