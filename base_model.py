from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score

class BaseTurnoverModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model.
        X: pandas DataFrame or numpy array of features
        y: pandas Series or numpy array of target labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probability of turnover (class 1).
        X: features
        Returns: numpy array of probabilities
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """
        Return feature importances or coefficients.
        Returns: dict or pandas Series
        """
        pass

    def evaluate(self, X, y):
        """
        Custom evaluation method.
        Calculates:
        - Recall @ Top 20%
        - F1-Score (Target > 0.65)
        - AUC-ROC (Target > 0.75)
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # AUC-ROC
        auc = roc_auc_score(y, y_pred_proba)
        
        # F1-Score
        f1 = f1_score(y, y_pred)
        
        # Recall @ Top 20%
        # 1. Create a dataframe with true labels and predicted probs
        results = pd.DataFrame({'y_true': y, 'y_prob': y_pred_proba})
        
        # 2. Sort by probability descending
        results = results.sort_values(by='y_prob', ascending=False)
        
        # 3. Take top 20% of the population
        top_20_cutoff = int(len(results) * 0.2)
        top_20_segment = results.iloc[:top_20_cutoff]
        
        # 4. Calculate how many actual churners are in this top 20%
        # The requirement is: "Recall@Top20%: The system requirement is to capture 70% of leavers within the top 20% of risk scores."
        # This technically means Lift or Recall at fixed depth. 
        # Recall = TP / (TP + FN) (Total positives)
        # Recall@20% = (Positives in Top 20%) / (Total Positives in entire dataset)
        
        total_positives = results['y_true'].sum()
        captured_positives = top_20_segment['y_true'].sum()
        
        recall_top_20 = captured_positives / total_positives if total_positives > 0 else 0.0
        
        metrics = {
            "Recall@Top20%": recall_top_20,
            "F1_Score": f1,
            "AUC_ROC": auc
        }
        
        return metrics
