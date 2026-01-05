import pandas as pd
import numpy as np
import os
from config import ENGLISH_TO_HEBREW, COLUMN_MAPPING

def generate_hilan_data(n_samples=2000, random_seed=42):
    np.random.seed(random_seed)
    
    data = {}
    
    # --- Demographics ---
    data['Tenure_Months'] = np.random.randint(1, 180, n_samples)
    data['Age'] = np.random.randint(22, 67, n_samples)
    data['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
    data['Marital_Status'] = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples, p=[0.3, 0.5, 0.15, 0.05])
    data['Residence_City'] = np.random.choice(['Tel Aviv', 'Haifa', 'Jerusalem', 'Rishon LeZion', 'Petah Tikva'], n_samples)
    data['Commute_Time_Est'] = np.random.normal(45, 20, n_samples).clip(10, 180)
    data['Organizational_Unit'] = np.random.choice(['R&D', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples)
    data['Job_Rank'] = np.random.choice(['Junior', 'Senior', 'Team Lead', 'Manager'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    data['Job_Rating'] = np.random.randint(1, 6, n_samples) # 1-5 rating
    data['Agreement_Type'] = np.random.choice(['Personal', 'Collective'], n_samples)

    # --- Flags (Risk Factors) ---
    # Logic: Higher probability of flag if certain conditions met (to make it realistic)
    data['Flag_Salary_Freeze'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) 
    data['Flag_Role_Stagnation'] = np.where(data['Tenure_Months'] > 36, np.random.choice([0, 1], n_samples, p=[0.6, 0.4]), 0)
    data['Flag_Manager_Change'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['Flag_Personal_Change'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    data['Flag_Annual_Hours_Change'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    data['Flag_Sick_Abuse'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    data['Flag_Sick_Abuse_1'] = data['Flag_Sick_Abuse'] # Duplicate?
    data['Flag_Actual_Hours'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

    # --- Stats ---
    data['Salary_Avg_12m'] = np.random.normal(15000, 5000, n_samples).clip(6000, 50000)
    data['Salary_StDev'] = data['Salary_Avg_12m'] * np.random.uniform(0.01, 0.1, n_samples)
    data['Salary_Median'] = data['Salary_Avg_12m'] # Approx
    data['Salary_Change_Value'] = np.where(data['Flag_Salary_Freeze'] == 1, 0, np.random.normal(1000, 500, n_samples).clip(0, 5000))
    
    data['Sick_Days_Avg'] = np.random.poisson(3, n_samples)
    data['Sick_Days_StDev'] = np.random.uniform(0, 2, n_samples)
    data['Sick_Days_Median'] = data['Sick_Days_Avg']
    
    data['Work_Load_Avg'] = np.random.normal(100, 10, n_samples) # Hours? %?
    data['Work_Load_StDev'] = np.random.uniform(0, 10, n_samples)
    data['Work_Load_Median'] = data['Work_Load_Avg']
    
    data['Manager_Count_Total'] = np.random.randint(1, 5, n_samples)

    # --- Target Generation (Logic) ---
    # Base probability
    prob = 0.1
    
    # Risk factors increase probability
    prob += np.where(data['Flag_Salary_Freeze'] == 1, 0.15, 0)
    prob += np.where(data['Flag_Role_Stagnation'] == 1, 0.15, 0)
    prob += np.where(data['Flag_Manager_Change'] == 1, 0.1, 0)
    prob += np.where(data['Commute_Time_Est'] > 60, 0.1, 0)
    prob += np.where(data['Job_Rating'] < 3, 0.1, 0)
    prob += np.where(data['Salary_Avg_12m'] < 8000, 0.1, 0)
    
    # Clip probability
    prob = np.clip(prob, 0, 0.9)
    
    data['Target_Churn'] = np.random.binomial(1, prob)
    
    # --- Survival Time (Months Until Event) ---
    # If Churn=1, time is between 1 and 12 months (since it's a 12m snapshot)
    # If Churn=0, time is > 12 (censored)
    months = np.zeros(n_samples)
    for i in range(n_samples):
        if data['Target_Churn'][i] == 1:
            months[i] = np.random.randint(1, 13)
        else:
            months[i] = np.random.randint(13, 36) # Censored, but for simulation we give a future date
            
    # We add this column even if not in Excel, or we can map it to something?
    # The Excel had 'אינדיקציה לעזיבה - המשתנה התלוי' which is Target_Churn.
    # It didn't have a time column. We will add 'Months_Until_Event' as an extra for now.
    data['Months_Until_Event'] = months

    df = pd.DataFrame(data)
    
    # Rename to Hebrew
    # We need to handle the extra column 'Months_Until_Event' which is not in mapping
    # We will keep it in English or drop it if strictly following Excel. 
    # But we need it for the project. Let's keep it as 'Months_Until_Event'.
    
    rename_map = {k: v for k, v in ENGLISH_TO_HEBREW.items() if k in df.columns}
    df_hebrew = df.rename(columns=rename_map)
    
    return df_hebrew

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    df = generate_hilan_data(n_samples=5000)
    output_path = 'data/raw/hilan_synthetic_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Generated {len(df)} samples to {output_path}")
    print("Columns:", df.columns.tolist())
