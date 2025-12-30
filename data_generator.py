import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000, random_seed=42):
    """
    Generates a mock Pandas DataFrame for Turnover Prediction.
    
    Features:
    - Target: Target_Churn (0/1), Months_Until_Event (Numeric)
    - Demographics: Tenure_Months, Age, Gender, Marital_Status, Commute_Time
    - Calculated Flags: Flag_Salary_Freeze, Flag_Role_Stagnation, Flag_Manager_Change, Flag_Workload_Drop
    - Stats: Salary_Avg_12m, Sick_Days_Avg, Work_Load_Avg
    """
    np.random.seed(random_seed)
    
    data = {}
    
    # Demographics
    data['Tenure_Months'] = np.random.randint(1, 120, n_samples)
    data['Age'] = np.random.randint(22, 65, n_samples)
    data['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
    data['Marital_Status'] = np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
    data['Commute_Time'] = np.random.normal(45, 15, n_samples).clip(10, 180)
    
    # Calculated Flags
    data['Flag_Salary_Freeze'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 1 if no raise in 24m
    data['Flag_Role_Stagnation'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) # 1 if no promo in 36m
    data['Flag_Manager_Change'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15]) # 1 if changed in 12m
    data['Flag_Workload_Drop'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Stats
    data['Salary_Avg_12m'] = np.random.normal(7000, 2000, n_samples).clip(3000, 20000)
    data['Sick_Days_Avg'] = np.random.poisson(2, n_samples)
    data['Work_Load_Avg'] = np.random.normal(0.8, 0.1, n_samples).clip(0, 1.2)
    
    # Generate Target based on some logic to make it learnable
    # Higher churn probability if: Low Salary, High Commute, Role Stagnation, Salary Freeze
    
    base_prob = 0.1
    prob_adj = (
        (data['Flag_Salary_Freeze'] * 0.3) +
        (data['Flag_Role_Stagnation'] * 0.3) +
        (data['Flag_Manager_Change'] * 0.1) +
        ((data['Commute_Time'] > 60).astype(int) * 0.2) +
        ((data['Salary_Avg_12m'] < 5000).astype(int) * 0.2) -
        (data['Tenure_Months'] > 60).astype(int) * 0.2
    )
    
    final_prob = (base_prob + prob_adj).clip(0, 1)
    
    data['Target_Churn'] = np.random.binomial(1, final_prob)
    
    # Time-to-Event (Survival Analysis)
    # If Churn=1, event happened within 12 months (or some window). 
    # Let's say max observation window is 120 months (10 years)
    # If Churn=0, censored at T=12 (Snapshot strategy T-12 months) or random large number?
    # Prompt says: "Target_Churn (0/1), Months_Until_Event (Numeric for survival analysis)"
    # "Snapshot Strategy: Assume the input data is already structured as a historical snapshot (T-12 months)."
    # This implies we are looking at data from 12 months ago.
    # If Target_Churn=1, it means they left within the last 12 months.
    # If Target_Churn=0, they are still here after 12 months.
    
    # We generate Months_Until_Event
    months = np.zeros(n_samples)
    for i in range(n_samples):
        if data['Target_Churn'][i] == 1:
            months[i] = np.random.randint(1, 13) # Left within 1-12 months
        else:
            months[i] = 12 # Censored at 12 months (didn't leave in the window)
            
    data['Months_Until_Event'] = months
    
    df = pd.DataFrame(data)
    
    # One-hot encoding for categorical variables if needed, OR we return raw and let models handle/pipeline handle it.
    # The prompt says "mix of categorical, numerical". Random Forest/XGBoost can handle some, but sklearn usually needs encoding.
    # For simplicity and robustness, let's one-hot encode categorical features here or assume pre-processing in model.
    # Given the request is for a library of model classes, it's better if the data generator returns raw-ish data 
    # but maybe pre-encoded for the baseline models which definitely need it.
    # Sklearn LogisticRegression needs numeric input.
    # Let's simple OHE the categorical columns: Gender, Marital_Status.
    
    df = pd.get_dummies(df, columns=['Gender', 'Marital_Status'], drop_first=True)
    
    # Ensure boolean/int types are clean
    return df
