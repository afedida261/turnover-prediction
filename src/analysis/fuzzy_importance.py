import numpy as np
import pandas as pd
from typing import Dict, List, Any

class FuzzyFeatureImportance:
    """
    Implements a Fuzzy Information Fusion approach for feature importance aggregation.
    Inspired by Rengasamy et al. (2022).
    
    This method addresses the instability and bias of single-model importance measures
    by fusing rankings from multiple models using fuzzy membership functions.
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.model_importances = {}
        
    def add_model_importance(self, model_name: str, importance_dict: Dict[str, float]):
        """
        Add importance scores from a model.
        Scores will be normalized to [0, 1] range.
        """
        # Ensure all features exist, fill missing with 0
        scores = np.array([abs(importance_dict.get(feat, 0.0)) for feat in self.feature_names])
        
        # Normalize to [0, 1] (Min-Max normalization)
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = scores
            
        self.model_importances[model_name] = normalized_scores
        
    def fuzzify(self, x: float) -> Dict[str, float]:
        """
        Apply fuzzy membership functions to a normalized score x in [0, 1].
        Returns memberships for Low, Medium, High importance.
        Using Trapezoidal/Triangular membership functions.
        """
        # Low Importance (Z-shape)
        if x <= 0.3:
            mu_low = 1.0
        elif 0.3 < x < 0.5:
            mu_low = (0.5 - x) / 0.2
        else:
            mu_low = 0.0
            
        # Medium Importance (Triangular)
        if x <= 0.2 or x >= 0.8:
            mu_med = 0.0
        elif 0.2 < x <= 0.5:
            mu_med = (x - 0.2) / 0.3
        elif 0.5 < x < 0.8:
            mu_med = (0.8 - x) / 0.3
            
        # High Importance (S-shape)
        if x <= 0.5:
            mu_high = 0.0
        elif 0.5 < x < 0.7:
            mu_high = (x - 0.5) / 0.2
        else:
            mu_high = 1.0
            
        return {'Low': mu_low, 'Medium': mu_med, 'High': mu_high}
    
    def calculate_consensus(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Calculate the Consensus Importance Score (CIS) through fuzzy fusion.
        
        Steps:
        1. Fuzzify each model's score for each feature.
        2. Aggregate fuzzy sets (Weighted Sum).
        3. Defuzzify to get a crisp consensus score.
        """
        if not self.model_importances:
            return pd.DataFrame()
            
        n_features = len(self.feature_names)
        n_models = len(self.model_importances)
        
        # Default equal weights if not provided
        if weights is None:
            weights = {name: 1.0/n_models for name in self.model_importances}
            
        consensus_scores = np.zeros(n_features)
        
        # Defuzzification centroids (Centroid method)
        # Low -> 0.15, Medium -> 0.5, High -> 0.85 (Approximations based on shapes)
        centroids = {'Low': 0.15, 'Medium': 0.5, 'High': 0.85}
        
        for i, feature in enumerate(self.feature_names):
            aggregated_fuzzy = {'Low': 0.0, 'Medium': 0.0, 'High': 0.0}
            
            for model_name, scores in self.model_importances.items():
                score = scores[i]
                fuzzy_vals = self.fuzzify(score)
                weight = weights.get(model_name, 1.0/n_models)
                
                # Aggregate (Weighted Sum aggregation)
                for level in fuzzy_vals:
                    aggregated_fuzzy[level] += fuzzy_vals[level] * weight
            
            # Defuzzify (Center of Gravity)
            numerator = sum(aggregated_fuzzy[level] * centroids[level] for level in centroids)
            denominator = sum(aggregated_fuzzy.values())
            
            if denominator > 0:
                consensus_scores[i] = numerator / denominator
            else:
                consensus_scores[i] = 0.0
                
        # Create result DataFrame
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'Consensus_Score': consensus_scores
        })
        
        # Add individual model scores for comparison (un-normalized for reference? No, use normalized for heatmaps)
        for model_name, scores in self.model_importances.items():
            results[f'{model_name}_Norm'] = scores
            
        return results.sort_values('Consensus_Score', ascending=False)
