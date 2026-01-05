# Column Mapping from Hebrew (Raw Data) to English (Internal)
COLUMN_MAPPING = {
    'אינדיקציה לעזיבה - המשתנה התלוי': 'Target_Churn',
    'ותק בחודשים': 'Tenure_Months',
    'גיל': 'Age',
    'מגדר': 'Gender',
    'מצב משפחתי': 'Marital_Status',
    'זמן נסיעה לעבודה': 'Commute_Time_Est',
    'יישוב מגורים': 'Residence_City',
    'סעיף תקציבי/שיוך ארגוני': 'Organizational_Unit',
    'מעמד': 'Job_Rank',
    'דירוג': 'Job_Rating',
    'דגל תפקיד': 'Flag_Role_Stagnation',
    'דגל מנהל': 'Flag_Manager_Change',
    'דגל ניצול מחלה': 'Flag_Sick_Abuse', # Assuming this based on name
    'דגל שעות עבודה': 'Flag_Annual_Hours_Change',
    'דגל שכר': 'Flag_Salary_Freeze',
    'דגל ניצול מחלה.1': 'Flag_Sick_Abuse_1', # Duplicate in excel?
    'דגל מצב אישי': 'Flag_Personal_Change',
    'avg_Payment': 'Salary_Avg_12m',
    'stdevp_Payment': 'Salary_StDev',
    'Median_Payment': 'Salary_Median',
    'שינוי השכר בשקלים': 'Salary_Change_Value',
    'avg_illness': 'Sick_Days_Avg',
    'stdevp_illness': 'Sick_Days_StDev',
    'Median_illness': 'Sick_Days_Median',
    'avg_omes': 'Work_Load_Avg',
    'stdevp_omes': 'Work_Load_StDev',
    'Median_omes': 'Work_Load_Median',
    'count_managers': 'Manager_Count_Total',
    # Additional columns that might be in the excel but not mapped yet
    'דגל שעות עבודה בפועל': 'Flag_Actual_Hours',
    'הסכם': 'Agreement_Type'
}

# Reverse mapping for generation
ENGLISH_TO_HEBREW = {v: k for k, v in COLUMN_MAPPING.items()}

# Features to use in model
NUMERIC_FEATURES = [
    'Tenure_Months', 'Age', 'Commute_Time_Est', 
    'Salary_Avg_12m', 'Salary_StDev', 'Salary_Median', 'Salary_Change_Value',
    'Sick_Days_Avg', 'Sick_Days_StDev', 'Sick_Days_Median',
    'Work_Load_Avg', 'Work_Load_StDev', 'Work_Load_Median',
    'Manager_Count_Total'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Marital_Status', 'Residence_City', 'Organizational_Unit', 
    'Job_Rank', 'Job_Rating', 'Agreement_Type'
]

BINARY_FEATURES = [
    'Flag_Role_Stagnation', 'Flag_Manager_Change', 'Flag_Annual_Hours_Change',
    'Flag_Salary_Freeze', 'Flag_Personal_Change', 'Flag_Actual_Hours'
]

TARGET_COL = 'Target_Churn'
