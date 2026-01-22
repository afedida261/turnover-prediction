# Configuration for Real HR Data (first_file.xlsx)

# Column definitions for the real data
TARGET_COL = 'leave_ind'  # 1 = left in last year, 0 = stayed

# ID columns - to be excluded from features
ID_COLUMNS = ['fictive-ovedmiun', 'fictive2']

# Output column - should be replaced by model predictions
OUTPUT_COLUMN = 'turnover_prob'

# Columns to exclude from features (IDs + target + output)
EXCLUDE_COLUMNS = ID_COLUMNS + [TARGET_COL, OUTPUT_COLUMN]

# Numeric features from the real data
NUMERIC_FEATURES = [
    'vetek_months',        # Tenure in months
    'age',                 # Age
    'distance_to_work',    # Distance to work (estimated)
    'Semel_Yishuv',        # City code
    'Seif',                # Budget section
    'Maamad',              # Job status/rank
    'tafkidCode',          # Role code
    'manager_Code',        # Manager code
    'Tafkid',              # Role flag
    'Maneger',             # Manager flag
    'Mahala',              # Department flag
    'Distance',            # Distance flag
    'WorkHours',           # Work hours flag
    'Sahar',               # Salary flag
    'MZV_Flag',            # MZV flag
    'hedrut',              # Absence flag
    'children',            # Children flag
    'avg_Payment',         # Average payment over 12 months
    'stdevp_Payment',      # Std deviation of payment
    'Median_Payment',      # Median payment
    'change_in_salary_bySHKL', # Salary change in shekels
    'avg_illness',         # Average sick days
    'stdevp_illness',      # Std deviation of sick days
    'Median_illness',      # Median sick days
    'avg_omes',            # Average workload
    'stdevp_omes',         # Std deviation of workload
    'Median_omes',         # Median workload
    'count_managers',      # Number of managers
]

# Categorical features
CATEGORICAL_FEATURES = [
    'TeurGroupHscm',           # Employment type (חודשי/גלובלי/שעתי)
    'gender',                  # Gender (גבר/אישה)
    'EMP_Matzav_Mishpachti',   # Marital status
    'Yishuv',                  # City name
]

# Binary/Flag features (already numeric 0/1)
BINARY_FEATURES = [
    'Tafkid',      # Role change flag
    'Maneger',     # Manager change flag
    'Mahala',      # Department change flag
    'Distance',    # Distance change flag
    'WorkHours',   # Work hours change flag
    'Sahar',       # Salary freeze flag
    'MZV_Flag',    # MZV flag
    'hedrut',      # Absence flag
    'children',    # Children change flag
]

# Feature name mapping (Hebrew descriptions for interpretability)
FEATURE_DESCRIPTIONS = {
    # Original features
    'vetek_months': 'Tenure (Months)',
    'age': 'Age',
    'distance_to_work': 'Estimated Commute Distance',
    'TeurGroupHscm': 'Employment Type',
    'gender': 'Gender',
    'EMP_Matzav_Mishpachti': 'Marital Status',
    'Yishuv': 'City of Residence',
    'Semel_Yishuv': 'City Code',
    'Seif': 'Budget Section',
    'Maamad': 'Job Rank',
    'tafkidCode': 'Role Code',
    'manager_Code': 'Manager Code',
    'Tafkid': 'Role Change Flag',
    'Maneger': 'Manager Change Flag',
    'Mahala': 'Department Change Flag',
    'Distance': 'Distance Change Flag',
    'WorkHours': 'Work Hours Change Flag',
    'Sahar': 'Salary Freeze Flag',
    'MZV_Flag': 'MZV Flag',
    'hedrut': 'Absence Flag',
    'children': 'Children Change Flag',
    'avg_Payment': 'Avg Salary (12m)',
    'stdevp_Payment': 'Salary Std Dev',
    'Median_Payment': 'Median Salary',
    'change_in_salary_bySHKL': 'Salary Change (NIS)',
    'avg_illness': 'Avg Sick Days',
    'stdevp_illness': 'Sick Days Std Dev',
    'Median_illness': 'Median Sick Days',
    'avg_omes': 'Avg Workload',
    'stdevp_omes': 'Workload Std Dev',
    'Median_omes': 'Median Workload',
    'count_managers': 'Number of Managers',
    
    # Time-series aggregated features
    'vetek_months_mean': 'Tenure Avg (Months)',
    'vetek_months_std': 'Tenure Variability',
    'vetek_months_trend': 'Tenure Trend',
    'age_mean': 'Age Avg',
    'age_std': 'Age Variability',
    'age_trend': 'Age Trend',
    'avg_Payment_mean': 'Salary Avg (Mean)',
    'avg_Payment_std': 'Salary Avg Variability',
    'avg_Payment_trend': 'Salary Trend',
    'stdevp_Payment_mean': 'Salary StdDev Avg',
    'stdevp_Payment_std': 'Salary StdDev Variability',
    'Median_Payment_mean': 'Median Salary Avg',
    'Median_Payment_std': 'Median Salary Variability',
    'avg_illness_mean': 'Sick Days Avg (Mean)',
    'avg_illness_std': 'Sick Days Variability',
    'avg_illness_trend': 'Sick Days Trend',
    'avg_omes_mean': 'Workload Avg (Mean)',
    'avg_omes_std': 'Workload Variability',
    'avg_omes_trend': 'Workload Trend',
    'stdevp_omes_mean': 'Workload StdDev Avg',
    'stdevp_omes_std': 'Workload StdDev Variability',
    'Median_omes_mean': 'Median Workload Avg',
    'Median_omes_std': 'Median Workload Variability',
    'manager_Code_mean': 'Manager Code Avg',
    'tafkidCode_mean': 'Role Code Avg',
    'count_managers_mean': 'Managers Count Avg',
    'count_managers_std': 'Managers Count Variability',
    'count_managers_trend': 'Managers Count Trend',
    'num_periods': 'Number of Time Periods',
    
    # Engineered features
    'salary_skewness': 'Salary Distribution Skewness',
    'salary_cv': 'Salary Coefficient of Variation',
    'salary_change_pct': 'Salary Change (%)',
    'workload_stability': 'Workload Stability Index',
    'workload_skewness': 'Workload Distribution Skewness',
    'illness_variability': 'Sick Days Variability Index',
    'career_start_age': 'Career Start Age',
    'tenure_ratio': 'Tenure Ratio (% of Career)',
    'tenure_years': 'Tenure (Years)',
    'is_new_employee': 'Is New Employee (<1 yr)',
    'is_senior': 'Is Senior Employee (>10 yrs)',
    'is_young': 'Is Young (<30)',
    'is_pre_retirement': 'Is Pre-Retirement (>55)',
    'manager_change_rate': 'Manager Change Rate (per year)',
    'workload_pay_ratio': 'Workload-to-Pay Ratio',
    'long_commute': 'Has Long Commute (>45 min)',
    'data_maturity': 'Data Maturity Score',
}

# Thresholds for outlier detection (using IQR multiplier)
OUTLIER_IQR_MULTIPLIER = 3.0

# Columns with known extreme value issues
EXTREME_VALUE_COLUMNS = [
    'avg_Payment',      # Has values up to 90,000
    'stdevp_Payment',   # Has extreme std dev
    'Median_Payment',   # Has values up to 90,000
    'change_in_salary_bySHKL',  # Has extreme negative/positive values
    'avg_illness',      # Has values up to 72
]
