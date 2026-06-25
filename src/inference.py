import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

# Assuming this file is under src/
from src.data_loader import RealExcelDataLoader
from src.config import ID_COLUMNS, TARGET_COL, OUTPUT_COLUMN
from src.datasets import DatasetSpec
from src.static_preprocessing import build_static_model_frame

class TurnoverInferenceAPI:
    def __init__(self, pipeline_path: str = "artifacts/model_pipeline.pkl"):
        """Initialize the inference API by loading the saved pipeline."""
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline artifact not found at {pipeline_path}. Please run main.py first.")
            
        self.pipeline = joblib.load(pipeline_path)
        self.mode = self.pipeline.get('mode', 'legacy')

        if self.pipeline.get('candidate') and self.pipeline.get('feature_columns') and self.pipeline.get('pipeline') is not None:
            self.mode = 'final_eda'
            self.model = self.pipeline['pipeline']
            self.scaler = None
            self.feature_names = list(self.pipeline['feature_columns'])
            self.decision_threshold = float(self.pipeline.get('decision_threshold', 0.5))
            self.employee_id_col = 'fictive_employee'
            self.time_col = 'calc_month'
            return

        self.model = self.pipeline['model']
        self.scaler = self.pipeline.get('scaler')
        self.feature_names = self.pipeline.get('feature_names') or self.pipeline.get('selected_features')
        self.decision_threshold = 0.5
        self.employee_id_col = self.pipeline.get('dataset_config', {}).get('employee_id_col', 'fictive2')
        self.time_col = self.pipeline.get('dataset_config', {}).get('time_col')
        
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

        if self.mode == "final_eda":
            return self._predict_final_eda(employee_df)

        if self.mode == "static_clean":
            return self._predict_static_clean(employee_df)
            
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


    @staticmethod
    def risk_category(prob: float) -> str:
        if prob <= 0.3:
            return 'Low Risk'
        if prob <= 0.5:
            return 'Medium Risk'
        if prob <= 0.7:
            return 'High Risk'
        return 'Very High Risk'

    def _predict_final_eda(self, employee_df: pd.DataFrame) -> Tuple[float, str]:
        frame = employee_df.copy()
        if self.time_col and self.time_col in frame.columns:
            frame = frame.sort_values(self.time_col)
        latest = frame.tail(1).copy()
        if latest.empty:
            raise ValueError("No employee rows available for final-model inference.")

        for column in self.feature_names:
            if column not in latest.columns:
                latest[column] = pd.NA
        X = latest[self.feature_names]
        probs = self.model.predict_proba(X)
        prob = float(probs[0][1] if hasattr(probs[0], '__getitem__') else probs[0])
        return prob, self.risk_category(prob)

    def _predict_static_clean(self, employee_df: pd.DataFrame) -> Tuple[float, str]:
        dataset_config = self.pipeline.get('dataset_config', {})
        spec = DatasetSpec(
            tag=self.pipeline.get('dataset_tag', 'unknown'),
            path=self.pipeline.get('source_path', ''),
            employee_id_col=dataset_config.get('employee_id_col', 'fictive2'),
            time_col=dataset_config.get('time_col'),
            header_row=dataset_config.get('header_row', 0),
        )

        frame, _ = build_static_model_frame(employee_df, spec)
        X = frame.drop(columns=[TARGET_COL], errors='ignore')
        X = X.drop(columns=[spec.employee_id_col], errors='ignore')

        selected_features = self.pipeline.get('selected_features') or self.feature_names
        for col in selected_features:
            if col not in X.columns:
                X[col] = pd.NA
        X = X[selected_features]

        probs = self.model.predict_proba(X)
        if hasattr(probs, 'ndim') and probs.ndim == 1:
            prob = float(probs[0])
        else:
            prob = float(probs[0][1] if hasattr(probs[0], '__getitem__') else probs[0])

        if prob <= 0.3:
            category = 'Low Risk'
        elif prob <= 0.5:
            category = 'Medium Risk'
        elif prob <= 0.7:
            category = 'High Risk'
        else:
            category = 'Very High Risk'

        return prob, category
