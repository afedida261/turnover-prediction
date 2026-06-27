import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Assuming this file is under src/
from src.data_loader import RealExcelDataLoader
from src.config import ID_COLUMNS, TARGET_COL, OUTPUT_COLUMN

class TurnoverInferenceAPI:
    def __init__(self, pipeline_path: str = "artifacts/model_pipeline_first_file_random.pkl"):
        """Initialize the inference API by loading the saved pipeline."""
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline artifact not found at {pipeline_path}. Please run main.py first.")
            
        self.pipeline = joblib.load(pipeline_path)
        self.model = self.pipeline['model']
        self.scaler = self.pipeline['scaler']
        self.feature_names = self.pipeline['feature_names']

        # The model may have been trained on a subset of features (due to
        # feature-group dropping in the workbench).  Detect the actual model
        # features so we can align at inference time.
        self.model_features = self._detect_model_features()

    def _detect_model_features(self):
        """Return the feature names the model actually expects."""
        model = self.model
        # scikit-learn estimators store feature_names_in_ after fit
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        # XGBoost Booster
        if hasattr(model, 'get_booster'):
            try:
                return model.get_booster().feature_names
            except Exception:
                pass
        # Ensemble / voting: check first estimator
        if hasattr(model, 'estimators_') and model.estimators_:
            first = model.estimators_[0]
            if hasattr(first, 'feature_names_in_'):
                return list(first.feature_names_in_)
        # Neural-net wrappers store n_features_in_ but not names;
        # fall back to the pipeline's saved list.
        return list(self.feature_names)

    def predict_risk(self, employee_df: pd.DataFrame) -> Tuple[float, str]:
        """
        Predict turnover risk for a given employee feature set.
        
        Args:
            employee_df (pd.DataFrame): DataFrame containing the employee's historical records.
                If it contains multiple rows, they must represent the same employee over time.
        
        Returns:
            Tuple[float, str]: The turnover probability (0 to 1) and Risk Category string.
        """
        if employee_df.empty:
            raise ValueError("Input DataFrame is empty.")
            
        # Instantiate loader to utilize its preprocessing methods
        loader = RealExcelDataLoader(filepath=None)
        
        # Inject the fitted scaler and expected feature names into the loader
        loader.scaler = self.scaler
        loader.feature_names = self.feature_names
        
        # Preprocess the DataFrame with is_inference=True
        # This will aggregate the time-series data, align features, and transform numerics.
        preprocessed_df = loader.preprocess(employee_df, is_inference=True)
        
        if preprocessed_df.empty:
            raise ValueError("Preprocessing resulted in an empty DataFrame.")
            
        # Align features: the model may expect fewer columns than the
        # preprocessor produces (feature groups may have been dropped during
        # training).  Add any missing columns as 0, drop extras.
        expected = self.model_features
        for col in expected:
            if col not in preprocessed_df.columns:
                preprocessed_df[col] = 0.0
        X = preprocessed_df[expected]
        
        # Predict probability
        probs = self.model.predict_proba(X)
        
        if hasattr(probs, 'ndim') and probs.ndim == 1:
            prob = float(probs[0])
        else:
            # Handle both list and numpy array 2D outputs
            prob = float(probs[0][1] if hasattr(probs[0], '__getitem__') else probs[0])
            
        
        # Determine risk category
        if prob <= 0.3:
            category = 'Low Risk'
        elif prob <= 0.5:
            category = 'Medium Risk'
        elif prob <= 0.7:
            category = 'High Risk'
        else:
            category = 'Very High Risk'
            
        return prob, category
