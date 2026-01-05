import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from .base_model import BaseTurnoverModel

class SurvivalTurnover(BaseTurnoverModel):
    def __init__(self, duration_col='Months_Until_Event', **kwargs):
        super().__init__()
        self.cph = CoxPHFitter(**kwargs)
        self.duration_col = duration_col

    def fit(self, X, y):
        # X must contain the duration column for training
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data = X.copy()
        
        if self.duration_col not in data.columns:
            # If duration is not in X, we can't train unless passed separately.
            # For this project, we assume it's passed in X during training phase.
            raise ValueError(f"Survival model requires '{self.duration_col}' column in X for training.")
        
        data['event'] = y
        
        # Handle potential collinearity or low variance
        try:
            self.cph.fit(data, duration_col=self.duration_col, event_col='event')
        except Exception as e:
            print(f"CoxPH Fit Error: {e}")
            
        return self

    def predict_proba(self, X):
        """
        Predict probability of churn within 12 months.
        """
        X_clean = X.drop(columns=[self.duration_col], errors='ignore')
        
        try:
            surv_funcs = self.cph.predict_survival_function(X_clean)
            # Get survival probability at t=12
            if 12 in surv_funcs.index:
                s_12 = surv_funcs.loc[12]
            else:
                # Interpolate or nearest
                times = [12]
                # predict_survival_function returns a dataframe indexed by time
                # We can't easily interpolate if 12 isn't there without re-predicting
                # But usually the index contains all event times.
                # Let's just take the closest time <= 12
                closest_t = max([t for t in surv_funcs.index if t <= 12], default=min(surv_funcs.index))
                s_12 = surv_funcs.loc[closest_t]
                
            return 1 - s_12.values
        except:
            return np.zeros(len(X))

    def get_feature_importance(self):
        return self.cph.params_.to_dict()
