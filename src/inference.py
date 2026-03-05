import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

# Assuming this file is under src/
from src.data_loader import RealExcelDataLoader
from src.config import ID_COLUMNS, TARGET_COL, OUTPUT_COLUMN

class TurnoverInferenceAPI:
    def __init__(self, pipeline_path: str = "artifacts/model_pipeline.pkl"):
        """Initialize the inference API by loading the saved pipeline."""
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline artifact not found at {pipeline_path}. Please run main.py first.")
            
        self.pipeline = joblib.load(pipeline_path)
        self.model = self.pipeline['model']
        self.scaler = self.pipeline['scaler']
        self.feature_names = self.pipeline['feature_names']
        
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
            
        # Extract features (should only be a single row because of time-series aggregation)
        X = preprocessed_df[self.feature_names]
        
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
