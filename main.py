import sys
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.generate_data import generate_hilan_data
from src.data_loader import SyntheticDataLoader, RealDataLoader
from src.models.classifiers import LogisticRegression, RandomForestTurnover, XGBoostTurnover, EnsembleTurnover
from src.models.nn_model import NeuralNetTurnover
from src.models.survival import SurvivalTurnover
from src.evaluator import Evaluator
from src.config import TARGET_COL

def main():
    parser = argparse.ArgumentParser(description="Turnover Prediction Pipeline")
    parser.add_argument('--mode', type=str, choices=['generate', 'real'], default='generate', 
                        help="Mode: 'generate' for synthetic data, 'real' for existing data")
    parser.add_argument('--data_path', type=str, default='data/raw/hilan_synthetic_data.csv',
                        help="Path to the data file")
    
    args = parser.parse_args()
    
    # 1. Data Handling
    if args.mode == 'generate':
        print(f"Mode: GENERATE. Using SyntheticDataLoader.")
        raw_data_path = args.data_path
        
        # Generate if not exists or forced (logic can be adjusted)
        if not os.path.exists(raw_data_path):
            print("Generating synthetic data...")
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            df_gen = generate_hilan_data(n_samples=5000)
            df_gen.to_csv(raw_data_path, index=False, encoding='utf-8-sig')
        else:
            print(f"Data already exists at {raw_data_path}")
            
        loader = SyntheticDataLoader(raw_data_path)
        
    else: # args.mode == 'real'
        print(f"Mode: REAL. Using RealDataLoader.")
        if not os.path.exists(args.data_path):
            print(f"Error: Data file not found at {args.data_path}")
            return
        
        loader = RealDataLoader(args.data_path)

    # 2. Loading & Preprocessing
    print("Loading and preprocessing data...")
    df = loader.load()
    df = loader.preprocess(df)
    
    # 3. Split Data
    # Check if Months_Until_Event exists
    duration_col = 'Months_Until_Event'
    if duration_col not in df.columns:
        print("Warning: Duration column not found. Survival model will fail or needs adjustment.")
    
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in data.")
        return

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 4. Model Definition
    # You can add new models here easily by adding to the dictionary
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestTurnover(),
        "XGBoost": XGBoostTurnover(use_label_encoder=False, eval_metric='logloss'),
        "Ensemble": EnsembleTurnover(),
        "Neural Network": NeuralNetTurnover(epochs=20, batch_size=64),
        "Survival CoxPH": SurvivalTurnover(duration_col=duration_col)
    }
    
    evaluator = Evaluator()
    
    print("\nTraining and Evaluating Models...")
    print("-" * 80)
    
    results = []
    
    for name, model in models.items():
        print(f"\nModel: {name}")
        
        # Handle Duration Column
        if name == "Survival CoxPH":
            # Survival model needs duration in X
            current_X_train = X_train
            current_X_test = X_test
        else:
            # Classification models must NOT see duration (leakage)
            current_X_train = X_train.drop(columns=[duration_col], errors='ignore')
            current_X_test = X_test.drop(columns=[duration_col], errors='ignore')
            
        # Fit
        try:
            model.fit(current_X_train, y_train)
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
            
        # Evaluate
        metrics = evaluator.evaluate(model, current_X_test, y_test)
        
        # Print
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
                
        metrics['Model'] = name
        results.append(metrics)

    # 5. Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    if results:
        results_df = pd.DataFrame(results)
        # Reorder columns
        cols = ['Model', 'Recall@Top20%', 'Requirement_Met', 'AUC_ROC', 'F1_Score']
        # Ensure cols exist
        cols = [c for c in cols if c in results_df.columns]
        print(results_df[cols].sort_values(by='Recall@Top20%', ascending=False).to_string(index=False))
    else:
        print("No models were trained successfully.")

if __name__ == "__main__":
    main()
