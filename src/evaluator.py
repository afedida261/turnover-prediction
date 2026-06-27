import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, log_loss

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, model, X_test, y_test, threshold=0.5):
        """
        Evaluates the model using standard metrics and the specific business metric:
        Recall @ Top 20% (Lift).
        """
        # Get probabilities
        try:
            y_prob = model.predict_proba(X_test)
            # If model returns (N, 2), take the second column
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
        except AttributeError:
            # Fallback for models that might not have predict_proba (e.g. SVM without probability=True)
            y_prob = model.predict(X_test)

        y_pred = (y_prob >= threshold).astype(int)
        
        # Standard Metrics
        metrics = {
            "AUC_ROC": roc_auc_score(y_test, y_prob),
            "F1_Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred)
        }
        
        # Business Metric: Recall @ Top 20%
        # 1. Create dataframe
        results = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob})
        
        # 2. Sort by probability descending
        results = results.sort_values(by='y_prob', ascending=False)
        
        # 3. Take top 20%
        top_20_count = int(len(results) * 0.2)
        top_20_segment = results.iloc[:top_20_count]
        
        # 4. Calculate Recall in this segment
        # How many churners did we catch in the top 20%?
        captured_churners = top_20_segment['y_true'].sum()
        total_churners = results['y_true'].sum()
        
        recall_at_top_20 = captured_churners / total_churners if total_churners > 0 else 0.0
        
        # Calculate Theoretical Max Recall @ Top 20%
        # max_captured = min(top_20_count, total_churners)
        # max_possible_recall = max_captured / total_churners
        max_possible_recall = min(top_20_count, total_churners) / total_churners if total_churners > 0 else 0.0
        
        # Calculate Precision @ Top 20% (Hit Rate in the top bucket)
        precision_at_top_20 = captured_churners / top_20_count if top_20_count > 0 else 0.0

        metrics["Recall@Top20%"] = recall_at_top_20
        metrics["Max_Recall@Top20%"] = max_possible_recall
        metrics["Precision@Top20%"] = precision_at_top_20
        
        # 5. Calculate Recall @ Top 50% (Coverage of likely churners)
        top_50_count = int(len(results) * 0.5)
        top_50_segment = results.iloc[:top_50_count]
        captured_50 = top_50_segment['y_true'].sum()
        recall_at_top_50 = captured_50 / total_churners if total_churners > 0 else 0.0
        metrics["Recall@Top50%"] = recall_at_top_50
        
        # Improvement Factor (WizWhy-style)
        # Primary probability = base rate of positive class in y_test
        # Expected error cost = log-loss when always predicting the base rate (baseline / no-skill)
        # Avg error cost = log-loss of actual model predictions
        # Improvement factor = expected_error_cost / avg_error_cost (WizWhy definition: > 1 is better)
        eps = 1e-15
        primary_prob = float(y_test.mean())
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        expected_error_cost = log_loss(y_test, np.full(len(y_test), np.clip(primary_prob, eps, 1 - eps)))
        avg_error_cost = log_loss(y_test, y_prob_clipped)
        improvement_factor = expected_error_cost / avg_error_cost if avg_error_cost > 0 else float('inf')

        metrics["Primary_Probability"] = primary_prob
        metrics["Avg_Error_Cost"] = avg_error_cost
        metrics["Error_Cost"] = expected_error_cost
        metrics["Improvement_Factor"] = improvement_factor

        return metrics
