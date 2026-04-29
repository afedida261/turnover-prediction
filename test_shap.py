import sys, os, traceback, io
import pandas as pd, numpy as np, joblib, shap

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from src.data_loader import RealExcelDataLoader
    from src.config import set_seed, FEATURE_DESCRIPTIONS

    set_seed(42)
    pipeline = joblib.load(r"artifacts/model_pipeline_first_file_preset.pkl")
    best_model = pipeline['model']
    scaler = pipeline['scaler']
    dataset_config = pipeline.get('dataset_config', {})
    feature_names = pipeline['feature_names']
    loader = RealExcelDataLoader(r"data/first_file/first_file.xlsx", **dataset_config)
    loader.scaler = scaler
    loader.feature_names = feature_names
    raw_df = loader.load()
    emp_col = dataset_config.get('employee_id_col', 'fictive2')
    if emp_col not in raw_df.columns:
        emp_col = 'fictive-oved' if 'fictive-oved' in raw_df.columns else emp_col
    dataset_config['employee_id_col'] = emp_col
    processed_df = loader.preprocess(raw_df, is_inference=True)
    
    X_shap = processed_df[feature_names].copy()
    
    test_data_path = r"data/first_file/test_data.xlsx"
    if os.path.exists(test_data_path):
        test_df = pd.read_excel(test_data_path)
        test_ids = set(test_df[emp_col].astype(float).astype(int).astype(str))
        kept_ids = [str(int(float(eid))) for eid in loader.get_kept_indices()]
        test_mask = [eid in test_ids for eid in kept_ids]
        X_shap = X_shap[test_mask].copy()

    X_shap = X_shap.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float64)

    readable_columns = [FEATURE_DESCRIPTIONS.get(c, c) for c in X_shap.columns]
    X_shap.columns = readable_columns

    raw_model = getattr(best_model, 'model', best_model)
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    
    if isinstance(raw_model, VotingClassifier):
        print("Using PermutationExplainer on VotingClassifier...")
        background = X_shap.sample(min(50, len(X_shap)), random_state=42)
        def proba_churn(X):
            # Model mathematically expects original english feature names to calculate!
            X_temp = X.copy()
            if isinstance(X_temp, pd.DataFrame):
                X_temp.columns = feature_names
            return raw_model.predict_proba(X_temp)[:, 1]
            
        explainer = shap.PermutationExplainer(proba_churn, background)
        raw_out = explainer(X_shap)
        print("Success! value shape:", raw_out.values.shape)
    else:
        print("TreeExplainer")

except Exception as e:
    traceback.print_exc()
