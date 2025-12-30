import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_generator import generate_synthetic_data
from classification_models import LogisticRegressionBaseline, RandomForestTurnover, XGBoostTurnover, EnsembleTurnover
from survival_model import SurvivalTurnover
from nn_model import NeuralNetTurnover

def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=2000)
    
    # Feature Engineering / Preprocessing
    # We need to separate features (X) and target (y)
    # Target for classification: Target_Churn
    # Target for survival: Target_Churn + Months_Until_Event
    
    # Define features
    target_col = 'Target_Churn'
    duration_col = 'Months_Until_Event'
    drop_cols = [target_col] # Keep duration in X for Survival model convenience, remove for classification if needed
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Models to train
    models = {
        "Logistic Regression": LogisticRegressionBaseline(),
        "Random Forest": RandomForestTurnover(),
        "XGBoost": XGBoostTurnover(use_label_encoder=False, eval_metric='logloss'),
        "Ensemble": EnsembleTurnover(),
        "Neural Network": NeuralNetTurnover(epochs=50, batch_size=32),
        "Survival CoxPH": SurvivalTurnover(duration_col=duration_col)
    }
    
    print("\nTraining and Evaluating Models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\nModel: {name}")
        
        # Training
        # For Classification models, we should ideally exclude 'Months_Until_Event' from X 
        # to avoid leakage if it's perfectly correlated or gives away the answer.
        # But 'Months_Until_Event' is usually not known at inference time?
        # "Snapshot Strategy: Assume the input data is already structured as a historical snapshot (T-12 months)."
        # At T-12, we DON'T know Months_Until_Event. That's the target for survival.
        # So we MUST DROP IT for Classification models.
        # Survival model NEEDS it for training (in X or separate).
        # My SurvivalTurnover.fit expects it in X.
        
        if name == "Survival CoxPH":
            # Pass full X (with duration)
            current_X_train = X_train
            current_X_test = X_test
        else:
            # Drop duration for classification to prevent leakage
            current_X_train = X_train.drop(columns=[duration_col], errors='ignore')
            current_X_test = X_test.drop(columns=[duration_col], errors='ignore')
        
        # Fit
        try:
             model.fit(current_X_train, y_train)
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

        # Evaluate
        # For evaluation, predict_proba in Survival model handles dropping the duration column internally if needed.
        # But base evaluate passes (X, y).
        
        metrics = model.evaluate(current_X_test, y_test)
        
        # Print metrics
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Feature Importance (first 5 for brevity)
        try:
            imp = model.get_feature_importance()
            if isinstance(imp, dict):
                sorted_imp = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                print(f"Top 5 Features: {sorted_imp}")
            else:
                print("Feature importance available (array)")
        except Exception as e:
            print(f"Could not get feature importance: {e}")

if __name__ == "__main__":
    main()
