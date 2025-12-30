import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from base_model import BaseTurnoverModel

class SurvivalTurnover(BaseTurnoverModel):
    def __init__(self, duration_col='Months_Until_Event', **kwargs):
        super().__init__()
        self.cph = CoxPHFitter(**kwargs)
        self.duration_col = duration_col

    def fit(self, X, y):
        """
        Fits CoxPHFitter.
        X: features (must include duration_col if not separate, but usually passed in X for convenience in this specific custom setup)
        y: target event (churn boolean)
        
        NOTE: lifelines requires a dataframe with both covariates and duration/event. 
        We will combine X and y.
        """
        # Ensure X is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data = X.copy()
        
        # Check if duration column is in data. 
        # If not, we cannot train a survival model unless we pass it separately, 
        # but the signature is fixed to fit(X, y).
        # We assume X contains the duration column 'Months_Until_Event'.
        if self.duration_col not in data.columns:
            raise ValueError(f"Survival model requires '{self.duration_col}' column in X for training.")
        
        # Add event column
        data['event'] = y
        
        # lifelines can have issues with low variance or collinearity, so we wrap in try/except or handle
        self.cph.fit(data, duration_col=self.duration_col, event_col='event')
        return self

    def predict_proba(self, X):
        """
        Predict probability of churn within 12 months.
        P(T <= 12) = 1 - S(12)
        """
        # Remove duration column from X if present, as predict doesn't need it (it's the target)
        X_clean = X.drop(columns=[self.duration_col], errors='ignore')
        
        # CoxPH predict_survival_function returns a DataFrame where index is time, columns are samples
        surv_funcs = self.cph.predict_survival_function(X_clean)
        
        # Get survival probability at t=12
        # If 12 is not in the index, we interpolate or take separate
        if 12 in surv_funcs.index:
            s_12 = surv_funcs.loc[12]
        else:
            # simple interpolation or closest
            # For simplicity, let's interpolate
            # But the index should be the event times observed in training.
            # We can use predict_survival_function at specific times?
            # It returns the function.
            # Let's try to get nearest or interpolate.
            # Actually, standard way:
            times = [12]
            s_12 = self.cph.predict_survival_function(X_clean, times=times).loc[12]
            
        return 1 - s_12.values

    def get_feature_importance(self):
        # Cox coefficients
        return self.cph.params_.to_dict()
    
    def evaluate(self, X, y):
        # We can implement specific survival metrics here like Concordance Index (c-index)
        # But to adhere to the base class interface and common metrics:
        return super().evaluate(X, y)
