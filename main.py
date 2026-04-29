import sys
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import RealExcelDataLoader
from src.models.classifiers import LogisticRegressionTurnover, RandomForestTurnover, XGBoostTurnover, EnsembleTurnover, AdaBoostTurnover
from src.models.nn_model import NeuralNetTurnover
from src.models.nn_advanced import RegularizedMLPTurnover
from src.evaluator import Evaluator
from src.config import TARGET_COL, FEATURE_DESCRIPTIONS, set_seed


class Reporter:
    """Writes lines to the results text file. Terminal stays clean for tqdm."""
    def __init__(self, path):
        self._file = open(path, 'w', encoding='utf-8')
        self.path = path

    def write(self, line=""):
        self._file.write(line + "\n")
        self._file.flush()

    def section(self, title):
        self.write("")
        self.write("=" * 80)
        self.write(title)
        self.write("=" * 80)

    def close(self):
        self._file.close()


def normalize_employee_id(value):
    """Normalize IDs for robust string matching across split files and datasets."""
    if pd.isna(value):
        return ""
    value_str = str(value).strip()
    try:
        value_float = float(value_str)
        if value_float.is_integer():
            return str(int(value_float))
    except ValueError:
        pass
    return value_str


def get_native_feature_importance(model, feature_names, top_n=15):
    """Fallback: extract and format native feature importance from model."""
    try:
        importance_dict = model.get_feature_importance()

        if isinstance(importance_dict, dict):
            mapped_dict = {}
            for key, value in importance_dict.items():
                if key.startswith('f') and key[1:].isdigit():
                    idx = int(key[1:])
                    if idx < len(feature_names):
                        mapped_dict[feature_names[idx]] = value
                    else:
                        mapped_dict[key] = value
                else:
                    mapped_dict[key] = value
            sorted_importance = sorted(mapped_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            sorted_importance = sorted(zip(feature_names, importance_dict), key=lambda x: abs(x[1]), reverse=True)

        return sorted_importance[:top_n]
    except Exception:
        return None


def compute_feature_importance(model, model_name, X, feature_names, top_n=15):
    """
    Prefer SHAP for interpretability. Fall back to the model's native
    feature importance for neural-net models (or if SHAP fails).
    Returns (list of (feature, score), method_label).
    """
    is_nn = 'Neural Net' in model_name or 'MLP' in model_name
    if not is_nn:
        try:
            from src.analysis.shap_explainability import compute_shap_summary
            shap_df = compute_shap_summary(model, X, feature_names, top_n=top_n)
            pairs = list(zip(shap_df['Feature'].tolist(), shap_df['Mean_Abs_SHAP'].tolist()))
            return pairs, "SHAP (mean |SHAP|)"
        except Exception as e:
            native = get_native_feature_importance(model, feature_names, top_n=top_n)
            if native is not None:
                return native, f"native (SHAP failed: {type(e).__name__})"
            return None, None

    native = get_native_feature_importance(model, feature_names, top_n=top_n)
    if native is not None:
        return native, "native (NN fallback)"
    return None, None


def bootstrap_predictive_uncertainty(model_builder, X_train, y_train, X_test,
                                     n_bootstrap=20, seed=42, progress_desc="Bootstrap"):
    """
    Bootstrap-based Bayesian-style predictive uncertainty.

    Refits the model on n_bootstrap bootstrap samples of the training data
    and collects predicted probabilities on X_test. Each bootstrap is an
    approximate draw from the posterior over models, so the spread across
    bootstraps approximates the posterior predictive distribution per sample.

    Returns: dict with mean, std, lo (2.5%), hi (97.5%) arrays of length n_test,
             plus summary stats.
    """
    rng = np.random.RandomState(seed)
    n = len(X_train)
    X_train_r = X_train.reset_index(drop=True)
    y_train_r = y_train.reset_index(drop=True)

    all_probs = []
    for _ in tqdm(range(n_bootstrap), desc=progress_desc, leave=False):
        idx = rng.choice(n, size=n, replace=True)
        X_b = X_train_r.iloc[idx]
        y_b = y_train_r.iloc[idx]
        m = model_builder()
        m.fit(X_b, y_b)
        p = m.predict_proba(X_test)
        if hasattr(p, 'ndim') and p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        all_probs.append(np.asarray(p).ravel())

    arr = np.vstack(all_probs)  # (B, n_test)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    ci_width = hi - lo

    # A prediction is "decisive" if the 95% credible interval stays on one
    # side of the 0.5 decision boundary.
    decisive = ((lo > 0.5) | (hi < 0.5)).mean() * 100

    return {
        'mean': mean,
        'std': std,
        'lo': lo,
        'hi': hi,
        'ci_width': ci_width,
        'n_bootstrap': n_bootstrap,
        'summary': {
            'mean_posterior_std': float(std.mean()),
            'median_posterior_std': float(np.median(std)),
            'mean_ci_width': float(ci_width.mean()),
            'median_ci_width': float(np.median(ci_width)),
            'decisive_pct': float(decisive),
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Turnover Prediction Pipeline")
    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to the Excel data file. If not provided, will prompt interactively.")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Path for output predictions file. Defaults to output/predictions_{dataset}_{split_type}.xlsx")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--results_path', type=str, default=None,
                        help="Path for the output results text file. Defaults to output/results_{dataset}.txt")

    args = parser.parse_args()

    if not args.data_path:
        data_dir = "data"
        if not os.path.exists(data_dir):
            print("Error: data/ directory not found!")
            import sys
            sys.exit(1)
            
        datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not datasets:
            print("No dataset directories found in data/.")
            import sys
            sys.exit(1)
            
        print("\n" + "="*40)
        print("Select Dataset for Training")
        print("="*40)
        for i, d in enumerate(datasets, 1):
            print(f"  {i}. {d}")
            
        while True:
            try:
                choice = input(f"\nEnter the number of the dataset to train on [1-{len(datasets)}]: ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(datasets):
                    selected_dataset = datasets[choice_idx]
                    dataset_folder = os.path.join(data_dir, selected_dataset)
                    raw_files = [f for f in os.listdir(dataset_folder) if not f.startswith("train_") and not f.startswith("test_") and f.endswith(".xlsx")]
                    if not raw_files:
                        print(f"Error: No raw Excel file found in {dataset_folder}")
                        import sys
                        sys.exit(1)
                    args.data_path = os.path.join(dataset_folder, raw_files[0])
                    break
                else:
                    print("Invalid choice, try again.")
            except ValueError:
                print("Please enter a valid number.")

    # Derive dataset tag dynamically based on the parent folder name
    dataset_tag = os.path.basename(os.path.dirname(args.data_path))

    # ----- 1. Data Handling -----
    dataset_dir = os.path.dirname(args.data_path)
    preset_files_exist = os.path.exists(os.path.join(dataset_dir, "train_data.xlsx")) and os.path.exists(os.path.join(dataset_dir, "test_data.xlsx"))
    
    using_preset_split = False
    if preset_files_exist:
        print("\n" + "="*40)
        print("Preset Split Files Found")
        print("="*40)
        print("  1. Use preset split (train_data.xlsx & test_data.xlsx)")
        print("  2. Force random 60/20/20 split")
        while True:
            choice = input(f"\nEnter your choice [1-2]: ").strip()
            if choice == '1':
                using_preset_split = True
                break
            elif choice == '2':
                using_preset_split = False
                break
            else:
                print("Invalid choice, try again.")

    split_type = 'preset' if using_preset_split else 'random'

    if not args.results_path:
        args.results_path = os.path.join('output', f'results_{dataset_tag}_{split_type}.txt')

    reporter = Reporter(args.results_path)
    report = reporter.write

    # Set seed for reproducibility
    set_seed(args.seed)
    
    reporter.section("EMPLOYEE TURNOVER PREDICTION")
    report(f"Dataset:          {dataset_tag}")
    report(f"Random seed:      {args.seed}")
    report(f"Split type:       {'preset (from data folder)' if using_preset_split else 'random 60/20/20'}")

    if not os.path.exists(args.data_path):
        reporter.close()
        print(f"Error: Data file not found at {args.data_path}")
        return

    dataset_config_map = {
        'first_file': {'employee_id_col': 'fictive2', 'time_col': 'fictive-ovedmiun'},
        'second_file': {'employee_id_col': 'fictive-oved', 'time_col': None},
        'factory_two': {'employee_id_col': 'fictive-oved', 'time_col': None},
    }
    dataset_config = dataset_config_map.get(dataset_tag, dataset_config_map['first_file'])

    if not args.output_path:
        args.output_path = os.path.join('output', f'predictions_{dataset_tag}_{split_type}.xlsx')

    # ----- 2. Loading & Preprocessing -----
    print(f"Loading and preprocessing {dataset_tag}...")
    loader = RealExcelDataLoader(args.data_path, **dataset_config)
    df_raw = loader.load()
    df = loader.preprocess(df_raw)
    feature_names = loader.get_feature_names()

    if TARGET_COL not in df.columns:
        reporter.close()
        print(f"Error: Target column '{TARGET_COL}' not found in data.")
        return

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    report("")
    report("Target distribution:")
    report(f"  Stayed (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    report(f"  Left   (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    # ----- 3. Split Data -----
    if using_preset_split:
        train_data_path = os.path.join(dataset_dir, "train_data.xlsx")
        test_data_path = os.path.join(dataset_dir, "test_data.xlsx")

        print(f"Using preset split from {dataset_dir}...")
        train_df = pd.read_excel(train_data_path)
        test_df = pd.read_excel(test_data_path)

        emp_col = dataset_config['employee_id_col']
        train_ids = set(normalize_employee_id(eid) for eid in train_df[emp_col])
        test_ids = set(normalize_employee_id(eid) for eid in test_df[emp_col])

        kept_ids = [normalize_employee_id(eid) for eid in loader.get_kept_indices()]
        train_mask = [eid in train_ids for eid in kept_ids]
        test_mask = [eid in test_ids for eid in kept_ids]

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        X_val, y_val = X_test, y_test
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=args.seed, stratify=y_temp
        )

    report("")
    report("Data split:")
    report(f"  Training:   {len(X_train):>6d}")
    report(f"  Validation: {len(X_val):>6d}")
    report(f"  Test:       {len(X_test):>6d}")

    # ----- 4. Model Builders -----
    # Use builders (callables) so we can re-instantiate for bootstrap uncertainty.
    scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
    model_builders = {
        "Logistic Regression": lambda: LogisticRegressionTurnover(class_weight='balanced', max_iter=1000),
        "Random Forest": lambda: RandomForestTurnover(n_estimators=200, class_weight='balanced', max_depth=15),
        "XGBoost": lambda: XGBoostTurnover(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            n_estimators=200,
        ),
        "AdaBoost": lambda: AdaBoostTurnover(n_estimators=200, learning_rate=0.5),
        "Ensemble": lambda: EnsembleTurnover(),
        "Neural Net": lambda: NeuralNetTurnover(epochs=100, batch_size=32, learning_rate=0.001),
        "Deep MLP (Regularized)": lambda: RegularizedMLPTurnover(epochs=150, batch_size=32, lr=0.001, n_folds=5),
    }

    evaluator = Evaluator()
    results = []
    best_model = None
    best_model_name = "None"
    best_auc = 0.0

    # Suppress noisy model-internal prints during training
    import io, contextlib

    pbar = tqdm(model_builders.items(), desc="Training models", unit="model")
    for name, builder in pbar:
        pbar.set_description(f"Training {name:25s}")
        try:
            model = builder()
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X_train, y_train)
        except Exception as e:
            tqdm.write(f"  [skip] {name}: {e}")
            continue

        val_metrics = evaluator.evaluate(model, X_val, y_val)
        test_metrics = evaluator.evaluate(model, X_test, y_test)

        if val_metrics['AUC_ROC'] > best_auc:
            best_auc = val_metrics['AUC_ROC']
            best_model = model
            best_model_name = name

        test_metrics['Model'] = name
        test_metrics['Val_AUC'] = val_metrics['AUC_ROC']
        test_metrics['Val_F1'] = val_metrics['F1_Score']
        test_metrics['Val_Precision'] = val_metrics['Precision']
        results.append(test_metrics)
    pbar.close()

    # ----- 5. Final results table -----
    reporter.section("MODEL COMPARISON")
    if results:
        results_df = pd.DataFrame(results)
        cols = ['Model', 'Val_AUC', 'AUC_ROC', 'Val_F1', 'F1_Score', 'Val_Precision', 'Precision',
                'Precision@Top20%', 'Recall@Top20%', 'Recall@Top50%',
                'Primary_Probability', 'Avg_Error_Cost', 'Error_Cost', 'Improvement_Factor']
                
        # If using preset split, Validation == Test, so hide redundant Validation metrics
        if split_type == 'preset':
            cols = [c for c in cols if not c.startswith('Val_')]
            
        cols = [c for c in cols if c in results_df.columns]
        display_df = results_df[cols].copy()
        rename_map = {
            'Model': 'Model',
            'Val_AUC': 'Val AUC',
            'AUC_ROC': 'Test AUC',
            'Val_F1': 'Val F1',
            'F1_Score': 'Test F1',
            'Val_Precision': 'Val Prec',
            'Precision': 'Test Prec',
            'Precision@Top20%': 'Prec@20%',
            'Recall@Top20%': 'Rec@20%',
            'Recall@Top50%': 'Rec@50%',
            'Primary_Probability': 'Base Rate',
            'Avg_Error_Cost': 'Avg Err',
            'Error_Cost': 'Err Cost',
            'Improvement_Factor': 'Impr Factor',
        }
        display_df = display_df.rename(columns=rename_map)
        
        sort_col = 'Val AUC' if 'Val AUC' in display_df.columns else 'Test AUC'
        display_df = display_df.sort_values(by=sort_col, ascending=False)

        # Format floats to 4 decimals for the text table
        float_cols = [c for c in display_df.columns if c != 'Model']
        for c in float_cols:
            display_df[c] = display_df[c].map(lambda v: f"{v:.4f}")

        report(display_df.to_string(index=False))
        report("")
        report(f"Best Model: {best_model_name} (Val AUC: {best_auc:.4f})")
    else:
        report("No models were trained successfully.")

    # ----- 6. Feature Importance (SHAP; fallback to native for NNs) -----
    reporter.section(f"FEATURE IMPORTANCE — {best_model_name}")
    if best_model is not None:
        print("Computing feature importance...")
        with contextlib.redirect_stdout(io.StringIO()):
            importance, method = compute_feature_importance(
                best_model, best_model_name, X_train, X.columns.tolist(), top_n=15
            )

        if importance:
            report(f"Method: {method}")
            report("")
            report(f"  {'Rank':<4}  {'Feature':<40}  {'Score':>12}")
            report("  " + "-" * 60)
            for i, (feat, imp) in enumerate(importance, 1):
                readable = FEATURE_DESCRIPTIONS.get(feat, feat) or feat
                if len(readable) > 38:
                    readable = readable[:35] + "..."
                report(f"  {i:<4d}  {readable:<40}  {imp:>12.4f}")
        else:
            report("Feature importance unavailable for this model.")

    # ----- 7. Bayesian-style Predictive Uncertainty (bootstrap posterior) -----
    reporter.section("PREDICTIVE UNCERTAINTY (bootstrap posterior)")
    if best_model is not None:
        print("Estimating predictive uncertainty via bootstrap...")
        with contextlib.redirect_stdout(io.StringIO()):
            unc = bootstrap_predictive_uncertainty(
                model_builders[best_model_name],
                X_train, y_train, X_test,
                n_bootstrap=20, seed=args.seed,
                progress_desc=None,
            )
        s = unc['summary']
        report("Method: refit the best model on B bootstrap resamples of the training")
        report("set; each refit is an approximate draw from the posterior over models.")
        report("The spread of predicted probabilities across refits approximates the")
        report("posterior predictive distribution at each test point.")
        report("")
        report(f"  B (bootstrap refits):                 {unc['n_bootstrap']}")
        report(f"  Test points:                          {len(unc['mean'])}")
        report(f"  Mean posterior std (per prediction):  {s['mean_posterior_std']:.4f}")
        report(f"  Median posterior std:                 {s['median_posterior_std']:.4f}")
        report(f"  Mean 95% CI width:                    {s['mean_ci_width']:.4f}")
        report(f"  Median 95% CI width:                  {s['median_ci_width']:.4f}")
        report(f"  Decisive predictions (95% CI one-sided of 0.5):  {s['decisive_pct']:.1f}%")

    # ----- 8. Generate output Excel + pickle -----
    if best_model is not None:
        y_prob_all = best_model.predict_proba(X)
        confidence_all = np.abs(y_prob_all - 0.5) * 2  # kept for Excel compatibility

        kept_employee_ids = loader.get_kept_indices()
        employee_id_col = dataset_config['employee_id_col']
        time_col = dataset_config['time_col']

        if time_col and time_col in df_raw.columns:
            output_df = (
                df_raw
                .sort_values([employee_id_col, time_col])
                .groupby(employee_id_col)
                .last()
                .reset_index()
            )
        else:
            output_df = df_raw.groupby(employee_id_col).last().reset_index()

        output_df = output_df[output_df[employee_id_col].isin(kept_employee_ids)].copy()
        output_df = output_df.sort_values(employee_id_col).reset_index(drop=True)

        pred_df = pd.DataFrame({
            employee_id_col: kept_employee_ids,
            'turnover_prob': y_prob_all,
            'prediction_confidence': confidence_all,
        })

        output_df = output_df.drop(columns=['turnover_prob'], errors='ignore')
        output_df = output_df.merge(pred_df, on=employee_id_col, how='left')

        output_df['risk_category'] = pd.cut(
            output_df['turnover_prob'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
        )

        output_column_mapping = {
            'leave_ind': 'Left Company (Actual)',
            employee_id_col: 'Employee ID',
            'turnover_prob': 'Turnover Probability',
            'prediction_confidence': 'Prediction Confidence',
            'risk_category': 'Risk Category',
            **FEATURE_DESCRIPTIONS,
        }
        if time_col and time_col in output_df.columns:
            output_column_mapping[time_col] = 'Record Index'

        output_df = output_df.rename(columns=output_column_mapping)
        output_df = output_df.sort_values('Turnover Probability', ascending=False)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        output_df.to_excel(args.output_path, index=False)

        reporter.section("OUTPUT FILES & RISK DISTRIBUTION")
        report(f"Predictions Excel: {args.output_path}")
        report(f"Unique employees:  {len(output_df)}")
        report("")
        report("Risk distribution:")
        risk_dist = output_df['Risk Category'].value_counts()
        for risk, count in risk_dist.items():
            report(f"  {risk:<16s} {count:>6d}  ({count/len(output_df)*100:.1f}%)")

        os.makedirs('artifacts', exist_ok=True)
        pipeline = {
            'model': best_model,
            'scaler': loader.get_scaler(),
            'feature_names': loader.get_feature_names(),
            'dataset_tag': dataset_tag,
            'dataset_config': dataset_config,
            'split_type': split_type,
            'use_preset_split': using_preset_split,
        }
        artifact_path = os.path.join('artifacts', f'model_pipeline_{dataset_tag}_{split_type}.pkl')
        joblib.dump(pipeline, artifact_path)
        report(f"Model pipeline:    {artifact_path}")

    report("")
    reporter.close()
    print(f"Done. Results saved to: {args.results_path}")

if __name__ == "__main__":
    main()
