import sys
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import RealExcelDataLoader
from src.models.classifiers import LogisticRegressionTurnover, RandomForestTurnover, XGBoostTurnover, EnsembleTurnover, AdaBoostTurnover
from src.models.nn_model import NeuralNetTurnover
from src.evaluator import Evaluator
from src.models.nn_model import NeuralNetTurnover
from src.evaluator import Evaluator
from src.config import TARGET_COL, FEATURE_DESCRIPTIONS, ID_COLUMNS, set_seed

def get_feature_importance(model, feature_names, top_n=15):
    """Extract and format feature importance from model."""
    try:
        importance_dict = model.get_feature_importance()
        
        if isinstance(importance_dict, dict):
            # Handle XGBoost's generic feature names (f0, f1, f2, etc.)
            # Map them back to actual feature names
            mapped_dict = {}
            for key, value in importance_dict.items():
                if key.startswith('f') and key[1:].isdigit():
                    # This is a generic XGBoost feature name like "f0", "f1"
                    idx = int(key[1:])
                    if idx < len(feature_names):
                        mapped_dict[feature_names[idx]] = value
                    else:
                        mapped_dict[key] = value
                else:
                    # Keep original key if it's not a generic name
                    mapped_dict[key] = value
            
            # Sort by importance value
            sorted_importance = sorted(mapped_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            # If array, convert to dict
            sorted_importance = sorted(zip(feature_names, importance_dict), key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_importance[:top_n]
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return None

def get_prediction_confidence(y_prob):
    """
    Calculate confidence metrics for predictions.
    Confidence is based on how far the probability is from 0.5 (uncertainty point).
    """
    # Convert to numpy if needed
    if hasattr(y_prob, 'values'):
        y_prob = y_prob.values
    
    # Confidence = distance from 0.5, scaled to 0-1
    confidence = np.abs(y_prob - 0.5) * 2
    
    return {
        'mean_confidence': np.mean(confidence),
        'median_confidence': np.median(confidence),
        'min_confidence': np.min(confidence),
        'max_confidence': np.max(confidence),
        'high_confidence_pct': np.mean(confidence > 0.7) * 100,  # % with >70% confidence
        'low_confidence_pct': np.mean(confidence < 0.3) * 100,   # % with <30% confidence
    }

def main():
    parser = argparse.ArgumentParser(description="Turnover Prediction Pipeline")
    parser.add_argument('--data_path', type=str, default='data/first_file.xlsx',
                        help="Path to the Excel data file")
    parser.add_argument('--output_path', type=str, default='output/predictions_output.xlsx',
                        help="Path for output predictions file")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--split_file', type=str, default=None,
                        help="Path to split directory containing train_ids.txt and test_ids.txt. "
                             "If provided, uses fixed employee-ID-based split instead of random split.")

    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # 1. Data Handling
    print("="*80)
    print("EMPLOYEE TURNOVER PREDICTION")
    print("="*80)
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    # 2. Loading & Preprocessing
    print("\n[1] Loading and preprocessing data...")
    print("-"*40)
    loader = RealExcelDataLoader(args.data_path)
    df_raw = loader.load()
    df = loader.preprocess(df_raw)
    
    # Get IDs for output
    original_ids = loader.get_original_ids()
    feature_names = loader.get_feature_names()
    
    # 3. Split Data
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in data.")
        return

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    print(f"\nTarget distribution:")
    print(f"  - Stayed (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  - Left (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    
    # Split into train/test
    if args.split_file:
        # --- Fixed split based on employee IDs from create_split.py ---
        train_ids_path = os.path.join(args.split_file, "train_ids.txt")
        test_ids_path = os.path.join(args.split_file, "test_ids.txt")

        if not os.path.exists(train_ids_path) or not os.path.exists(test_ids_path):
            print(f"Error: Could not find train_ids.txt / test_ids.txt in {args.split_file}")
            return

        with open(train_ids_path) as f:
            train_ids = set(line.strip() for line in f if line.strip())
        with open(test_ids_path) as f:
            test_ids = set(line.strip() for line in f if line.strip())

        # kept_employee_ids aligns 1-to-1 with X rows after aggregation
        kept_ids = [str(int(float(eid))) for eid in loader.get_kept_indices()]

        train_mask = [eid in train_ids for eid in kept_ids]
        test_mask  = [eid in test_ids  for eid in kept_ids]

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test  = X[test_mask]
        y_test  = y[test_mask]
        X_val, y_val = X_test, y_test  # use test as validation for best-model selection

        print(f"\nData Split (fixed, from {args.split_file}):")
        print(f"  - Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Test:     {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    else:
        # --- Default random 60/20/20 split ---
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=args.seed, stratify=y_temp
        )

        print(f"\nData Split (random 60/20/20):")
        print(f"  - Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  - Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Store test indices for output mapping
    test_indices = X_test.index
    
    # 4. Model Definition
    print("\n[2] Training and Evaluating Models...")
    print("-"*40)
    
    models = {
        "Logistic Regression": LogisticRegressionTurnover(class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestTurnover(n_estimators=200, class_weight='balanced', max_depth=15),
        "XGBoost": XGBoostTurnover(
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=len(y[y==0])/len(y[y==1]),  # Handle imbalance
            max_depth=6,
            n_estimators=200
        ),
        "AdaBoost": AdaBoostTurnover(n_estimators=200, learning_rate=0.5),
        "Ensemble": EnsembleTurnover(),
    }
    
    evaluator = Evaluator()
    results = []
    best_model = None
    best_auc = 0
    best_model_name = "None"
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Fit
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
        
        # Evaluate on Validation set
        val_metrics = evaluator.evaluate(model, X_val, y_val)
        print(f"  [Validation] AUC: {val_metrics['AUC_ROC']:.4f}, F1: {val_metrics['F1_Score']:.4f}")
        
        # Evaluate on Test set
        test_metrics = evaluator.evaluate(model, X_test, y_test)
        print(f"  [Test]       AUC: {test_metrics['AUC_ROC']:.4f}, F1: {test_metrics['F1_Score']:.4f}")
        print(f"  Recall@Top20%: {test_metrics['Recall@Top20%']:.4f} (Max: {test_metrics.get('Max_Recall@Top20%', 1.0):.4f})")
        print(f"  Recall@Top50%: {test_metrics.get('Recall@Top50%', 0.0):.4f}")
        print(f"  Precision@Top20%: {test_metrics.get('Precision@Top20%', 0.0):.4f}")
        print(f"  Requirement Met: {'✓ YES' if test_metrics['Requirement_Met'] else '✗ NO'}")
        print(f"    (Targets: Prec@20% >= 0.80, Recall@50% >= 0.85)")
        
        # Check for overfitting (validation vs test gap)
        auc_gap = abs(val_metrics['AUC_ROC'] - test_metrics['AUC_ROC'])
        if auc_gap > 0.05:
            print(f"  ⚠ Warning: AUC gap of {auc_gap:.4f} suggests possible overfitting")
        
        # Track best model based on validation AUC
        if val_metrics['AUC_ROC'] > best_auc:
            best_auc = val_metrics['AUC_ROC']
            best_model = model
            best_model_name = name
                
        test_metrics['Model'] = name
        test_metrics['Val_AUC'] = val_metrics['AUC_ROC']
        test_metrics['Val_F1'] = val_metrics['F1_Score']
        results.append(test_metrics)

    # 5. Feature Importance Analysis
    print("\n[3] Feature Importance Analysis")
    print("-"*40)

    if best_model is not None:
        print(f"\nTop features from best model ({best_model_name}) [Reference]:")
        importance = get_feature_importance(best_model, X.columns.tolist())
        
        if importance:
            print("\n  Rank | Feature                              | Importance")
            print("  " + "-"*60)
            for i, (feat, imp) in enumerate(importance, 1):
                # Get readable name if available
                readable = FEATURE_DESCRIPTIONS.get(feat, feat)
                if readable is None:
                    readable = feat
                # Truncate long feature names
                if len(readable) > 35:
                    readable = readable[:32] + "..."
                print(f"  {i:4d} | {readable:36s} | {imp:.4f}")
    
    # 6. Prediction Confidence
    print("\n[4] Prediction Confidence Analysis")
    print("-"*40)
    
    if best_model is not None:
        y_prob_test = best_model.predict_proba(X_test)
        confidence_metrics = get_prediction_confidence(y_prob_test)
        
        print(f"\n  Mean Confidence: {confidence_metrics['mean_confidence']*100:.1f}%")
        print(f"  Median Confidence: {confidence_metrics['median_confidence']*100:.1f}%")
        print(f"  High Confidence (>70%): {confidence_metrics['high_confidence_pct']:.1f}% of predictions")
        print(f"  Low Confidence (<30%): {confidence_metrics['low_confidence_pct']:.1f}% of predictions")
    
    # 7. Generate Output with Predictions
    print("\n[5] Generating Output File...")
    print("-"*40)
    
    if best_model is not None:
        # Get predictions for ALL processed data (not just test)
        y_prob_all = best_model.predict_proba(X)
        confidence_all = np.abs(y_prob_all - 0.5) * 2
        
        # Get the employee IDs that were kept after aggregation
        kept_employee_ids = loader.get_kept_indices()
        
        # Create output with most recent record per employee from raw data
        output_df = df_raw.sort_values(['fictive2', 'fictive-ovedmiun']).groupby('fictive2').last().reset_index()
        output_df = output_df[output_df['fictive2'].isin(kept_employee_ids)].copy()
        
        # Sort by employee ID to match the order of predictions
        output_df = output_df.sort_values('fictive2').reset_index(drop=True)
        
        # Create a mapping from employee ID to prediction
        pred_df = pd.DataFrame({
            'fictive2': kept_employee_ids,
            'turnover_prob': y_prob_all,
            'prediction_confidence': confidence_all
        })
        
        # Merge predictions with output
        output_df = output_df.drop(columns=['turnover_prob'], errors='ignore')
        output_df = output_df.merge(pred_df, on='fictive2', how='left')
        
        # Add risk category
        output_df['risk_category'] = pd.cut(
            output_df['turnover_prob'], 
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        )
        
        # Rename columns using FEATURE_DESCRIPTIONS for better readability
        # Create reverse mapping for output columns
        output_column_mapping = {
            'leave_ind': 'Left Company (Actual)',
            'fictive-ovedmiun': 'Record Index',
            'fictive2': 'Employee ID',
            'turnover_prob': 'Turnover Probability',
            'prediction_confidence': 'Prediction Confidence',
            'risk_category': 'Risk Category',
            **FEATURE_DESCRIPTIONS  # Include all feature descriptions
        }
        
        # Apply column renaming
        output_df = output_df.rename(columns=output_column_mapping)
        
        # Sort by turnover probability (highest risk first)
        output_df = output_df.sort_values('Turnover Probability', ascending=False)
        
        # Save to Excel
        output_df.to_excel(args.output_path, index=False)
        print(f"\n  Output saved to: {args.output_path}")
        print(f"  Total unique employees: {len(output_df)}")
        
        # Risk distribution
        print("\n  Risk Distribution:")
        risk_dist = output_df['Risk Category'].value_counts()
        for risk, count in risk_dist.items():
            print(f"    - {risk}: {count} ({count/len(output_df)*100:.1f}%)")

        # Save model pipeline artifact
        os.makedirs('artifacts', exist_ok=True)
        pipeline = {
            'model': best_model,
            'scaler': loader.get_scaler(),
            'feature_names': loader.get_feature_names(),
            'split_type': 'fixed' if args.split_file else 'random',
            'split_file': args.split_file
        }

        # Determine artifact filename based on split type
        if args.split_file:
            artifact_path = 'artifacts/model_pipeline_split.pkl'
        else:
            artifact_path = 'artifacts/model_pipeline_random.pkl'

        joblib.dump(pipeline, artifact_path)
        print(f"\n  Saved model pipeline artifact to {artifact_path}")

    # 8. Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    if results:
        results_df = pd.DataFrame(results)
        cols = ['Model', 'Val_AUC', 'AUC_ROC', 'Val_F1', 'F1_Score', 'Recall@Top20%', 'Requirement_Met']
        cols = [c for c in cols if c in results_df.columns]
        # Rename for display
        display_df = results_df[cols].copy()
        display_df.columns = ['Model', 'Val AUC', 'Test AUC', 'Val F1', 'Test F1', 'Recall@20%', 'Req Met']
        print(display_df.sort_values(by='Val AUC', ascending=False).to_string(index=False))
        
        print(f"\n✓ Best Model: {best_model_name} (Val AUC: {best_auc:.4f})")
    else:
        print("No models were trained successfully.")

if __name__ == "__main__":
    main()
