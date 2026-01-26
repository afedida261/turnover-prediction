from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from config import (
    TARGET_COL, EXCLUDE_COLUMNS, NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES, OUTLIER_IQR_MULTIPLIER, ID_COLUMNS,
    OUTPUT_COLUMN, FEATURE_DESCRIPTIONS
)

class BaseDataLoader(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.feature_names = None
        self.preprocessing_report = {}

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Loads the data from source."""
        pass

    def preprocess(self, df):
        """
        Basic preprocessing:
        - Handle missing values
        - Encode categorical variables
        """
        data = df.copy()
        
        # Drop duplicates if any
        data.drop_duplicates(inplace=True)
        
        # Fill missing values (simple strategy for now)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        for col in categorical_cols:
            if not data[col].mode().empty:
                data[col] = data[col].fillna(data[col].mode()[0])
            
        # One-Hot Encoding
        # We should define which columns to encode in config, but for now auto-detect
        # Exclude Target and Date columns if any
        cols_to_encode = [c for c in categorical_cols if c != TARGET_COL]
        
        if cols_to_encode:
            data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)
        
        return data

    def get_X_y(self, df, target_col=TARGET_COL):
        """Splits into features and target."""
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe.")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Also drop 'Months_Until_Event' from X if it exists, as it's a target for survival
        if 'Months_Until_Event' in X.columns:
            X = X.drop(columns=['Months_Until_Event'])
            
        return X, y


class RealExcelDataLoader(BaseDataLoader):
    """
    Data loader for the real HR data from first_file.xlsx
    Handles time-series aggregation, feature engineering, normalization, and preprocessing
    """
    
    def __init__(self, filepath):
        super().__init__(filepath)
        self.original_ids = None
        self.employee_id_col = 'fictive2'
        self.scaler = MinMaxScaler()
        self.numeric_columns_to_scale = []
        
    def load(self) -> pd.DataFrame:
        """Loads the Excel file."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_excel(self.filepath)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive preprocessing for real HR data:
        1. Aggregate time-series data per employee
        2. Feature engineering
        3. Handle missing values
        4. Handle outliers/extreme values
        5. Encode categorical variables
        6. Normalize numeric features (0-1 scaling)
        7. Remove non-feature columns
        """
        data = df.copy()
        self.preprocessing_report = {'original_rows': len(data)}
        
        # Store original IDs for output mapping
        self.original_ids = data[ID_COLUMNS].copy()
        
        # 1. Aggregate time-series data per employee
        print("\n[Preprocessing] Aggregating time-series data...")
        data = self._aggregate_time_series(data)
        
        # Store the employee IDs that we have after aggregation
        self.kept_employee_ids = data[self.employee_id_col].tolist()
        
        # 2. Feature Engineering - create new features BEFORE handling missing values
        print("[Preprocessing] Creating engineered features...")
        data = self._feature_engineering(data)
        
        # 3. Handle Missing Values
        data = self._handle_missing_values(data)
        
        # 4. Handle Outliers/Extreme Values
        data = self._handle_outliers(data)
        
        # 5. Encode Categorical Variables
        data = self._encode_categoricals(data)
        
        # 6. Remove non-feature columns (IDs, output column) but KEEP TARGET
        cols_to_drop = [c for c in EXCLUDE_COLUMNS if c in data.columns and c != TARGET_COL]
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # 7. Normalize numeric features (0-1 scaling)
        data = self._normalize_features(data)
        
        self.feature_names = [c for c in data.columns if c != TARGET_COL]
        self.preprocessing_report['final_features'] = len(self.feature_names)
        self.preprocessing_report['final_rows'] = len(data)
        
        print(f"\nPreprocessing complete: {len(data)} records, {len(self.feature_names)} features")
        
        return data
    
    def _aggregate_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple time periods per employee into single row with trend features.
        This handles the time-series nature where fictive2 identifies the same employee over time.
        """
        # Sort by employee and time index
        data = data.sort_values([self.employee_id_col, 'fictive-ovedmiun'])
        
        # Define aggregation strategy for each column type
        # Numeric columns: get last value, first value, mean, std, and trend (last - first)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ID_COLUMNS + [TARGET_COL, OUTPUT_COLUMN]]
        
        # Categorical columns: get the most recent value (last)
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        agg_dict = {}
        
        # For numeric columns, create multiple aggregations
        for col in numeric_cols:
            agg_dict[col] = ['last', 'first', 'mean', 'std']
        
        # For categorical columns, take the last value
        for col in cat_cols:
            agg_dict[col] = 'last'
        
        # Target: take the last value (most recent outcome)
        agg_dict[TARGET_COL] = 'last'
        
        # Group by employee and aggregate
        aggregated = data.groupby(self.employee_id_col).agg(agg_dict)
        
        # Flatten column names
        new_columns = []
        for col in aggregated.columns:
            if isinstance(col, tuple):
                if col[1] == 'last':
                    new_columns.append(col[0])  # Keep original name for 'last'
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
        
        aggregated.columns = new_columns
        aggregated = aggregated.reset_index()
        
        # Calculate trend features (change over time)
        for col in numeric_cols:
            first_col = f"{col}_first"
            last_col = col  # 'last' was renamed to original name
            if first_col in aggregated.columns and last_col in aggregated.columns:
                # Trend: positive means increasing, negative means decreasing
                aggregated[f"{col}_trend"] = aggregated[last_col] - aggregated[first_col]
                # Drop the 'first' column as we now have the trend
                aggregated.drop(columns=[first_col], inplace=True)
        
        # Count number of time periods per employee
        period_counts = data.groupby(self.employee_id_col).size().reset_index(name='num_periods')
        aggregated = aggregated.merge(period_counts, on=self.employee_id_col)
        
        print(f"  Aggregated {len(data)} records → {len(aggregated)} unique employees")
        print(f"  Created trend features for {len(numeric_cols)} numeric columns")
        
        self.preprocessing_report['unique_employees'] = len(aggregated)
        
        return aggregated
    
    def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create new meaningful features from existing ones.
        """
        engineered_features = []
        
        # 1. Salary-related features
        if 'avg_Payment' in data.columns and 'Median_Payment' in data.columns:
            # Salary skewness (positive = right-skewed, meaning occasional high payments)
            data['salary_skewness'] = (data['avg_Payment'] - data['Median_Payment']) / (data['avg_Payment'] + 1)
            engineered_features.append('salary_skewness')
        
        if 'stdevp_Payment' in data.columns and 'avg_Payment' in data.columns:
            # Salary coefficient of variation (stability indicator)
            data['salary_cv'] = data['stdevp_Payment'] / (data['avg_Payment'] + 1)
            engineered_features.append('salary_cv')
        
        if 'change_in_salary_bySHKL' in data.columns and 'avg_Payment' in data.columns:
            # Salary change as percentage of average salary
            data['salary_change_pct'] = data['change_in_salary_bySHKL'] / (data['avg_Payment'] + 1)
            engineered_features.append('salary_change_pct')
        
        # 2. Workload features
        if 'stdevp_omes' in data.columns and 'avg_omes' in data.columns:
            # Workload stability (lower is more stable)
            data['workload_stability'] = data['stdevp_omes'] / (data['avg_omes'] + 0.01)
            engineered_features.append('workload_stability')
        
        if 'avg_omes' in data.columns and 'Median_omes' in data.columns:
            # Workload skewness
            data['workload_skewness'] = (data['avg_omes'] - data['Median_omes']) / (data['avg_omes'] + 0.01)
            engineered_features.append('workload_skewness')
        
        # 3. Sick days features
        if 'stdevp_illness' in data.columns and 'avg_illness' in data.columns:
            # Sick days variability
            data['illness_variability'] = data['stdevp_illness'] / (data['avg_illness'] + 0.01)
            engineered_features.append('illness_variability')
        
        # 4. Age and tenure interactions
        if 'age' in data.columns and 'vetek_months' in data.columns:
            # Career start age (approximate)
            data['career_start_age'] = data['age'] - (data['vetek_months'] / 12)
            engineered_features.append('career_start_age')
            
            # Tenure relative to age (what % of working life spent here)
            # Assuming work starts at ~22
            data['tenure_ratio'] = data['vetek_months'] / ((data['age'] - 22).clip(lower=1) * 12)
            data['tenure_ratio'] = data['tenure_ratio'].clip(upper=1)  # Cap at 100%
            engineered_features.append('tenure_ratio')
        
        # 5. Experience level buckets
        if 'vetek_months' in data.columns:
            # Tenure in years for easier interpretation
            data['tenure_years'] = data['vetek_months'] / 12
            engineered_features.append('tenure_years')
            
            # New employee flag (< 1 year)
            data['is_new_employee'] = (data['vetek_months'] < 12).astype(int)
            engineered_features.append('is_new_employee')
            
            # Senior employee flag (> 10 years)
            data['is_senior'] = (data['vetek_months'] > 120).astype(int)
            engineered_features.append('is_senior')
        
        # 6. Age group flags
        if 'age' in data.columns:
            # Young employee (< 30)
            data['is_young'] = (data['age'] < 30).astype(int)
            engineered_features.append('is_young')
            
            # Pre-retirement (> 55)
            data['is_pre_retirement'] = (data['age'] > 55).astype(int)
            engineered_features.append('is_pre_retirement')
        
        # 7. Manager change impact (if count_managers exists)
        if 'count_managers' in data.columns and 'vetek_months' in data.columns:
            # Manager changes per year
            data['manager_change_rate'] = data['count_managers'] / (data['vetek_months'] / 12 + 0.1)
            engineered_features.append('manager_change_rate')
        
        # 8. Combined risk indicators
        # High workload + Low salary ratio
        if 'avg_omes' in data.columns and 'avg_Payment' in data.columns:
            # Normalize first for comparison
            omes_norm = (data['avg_omes'] - data['avg_omes'].min()) / (data['avg_omes'].max() - data['avg_omes'].min() + 0.01)
            payment_norm = (data['avg_Payment'] - data['avg_Payment'].min()) / (data['avg_Payment'].max() - data['avg_Payment'].min() + 0.01)
            data['workload_pay_ratio'] = omes_norm / (payment_norm + 0.01)
            engineered_features.append('workload_pay_ratio')
        
        # 9. Distance impact
        if 'distance_to_work' in data.columns:
            # Long commute flag (> 45 min)
            data['long_commute'] = (data['distance_to_work'] > 45).astype(int)
            engineered_features.append('long_commute')
        
        # 10. Use number of periods as stability indicator
        if 'num_periods' in data.columns:
            # More periods = more data = been tracked longer
            data['data_maturity'] = data['num_periods'] / data['num_periods'].max()
            engineered_features.append('data_maturity')
        
        print(f"  Created {len(engineered_features)} engineered features")
        self.preprocessing_report['engineered_features'] = engineered_features
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MinMax normalization (0-1 scaling) to all numeric features.
        This ensures all features are on the same scale for better model performance.
        """
        # Identify numeric columns to scale (exclude target and already binary columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target and columns that are already 0-1 (binary flags)
        binary_cols = [c for c in numeric_cols if data[c].nunique() <= 2]
        cols_to_scale = [c for c in numeric_cols if c != TARGET_COL and c not in binary_cols]
        
        self.numeric_columns_to_scale = cols_to_scale
        
        if cols_to_scale:
            # Fit and transform
            data[cols_to_scale] = self.scaler.fit_transform(data[cols_to_scale])
            print(f"[Preprocessing] Normalized {len(cols_to_scale)} numeric features to 0-1 scale")
        
        return data
    
    def get_kept_indices(self) -> List[int]:
        """Return the indices of rows that were kept after deduplication."""
        # Since we now aggregate, return based on employee IDs
        return self.kept_employee_ids
    
    def get_scaler(self) -> MinMaxScaler:
        """Return the fitted scaler for inverse transformations if needed."""
        return self.scaler
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies per column type."""
        missing_report = {}
        
        # Get columns with missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_count = data[col].isnull().sum()
            missing_pct = missing_count / len(data) * 100
            missing_report[col] = {'count': missing_count, 'pct': round(missing_pct, 2)}
            
            if col in EXCLUDE_COLUMNS or col == TARGET_COL:
                continue
                
            # Strategy based on missing percentage and column type
            if missing_pct > 50:
                # Too many missing - drop column
                print(f"  Dropping column '{col}' ({missing_pct:.1f}% missing)")
                data.drop(columns=[col], inplace=True)
            elif data[col].dtype in ['object'] or col == "TeurGroupHscm" or col == "Yishuv":
                # Categorical - fill with mode or 'Unknown'
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col] = data[col].fillna(mode_val[0])
                else:
                    data[col] = data[col].fillna('Unknown')
            else:
                # Numeric - fill with median
                print(f"  Filling missing values in numeric column '{col}' with median")
                data[col] = data[col].fillna(data[col].median())
        
        self.preprocessing_report['missing_values'] = missing_report
        print(f"Handled missing values in {len(missing_cols)} columns")
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR method - cap extreme values instead of removing.
        This preserves sample size while reducing impact of extreme values.
        """
        outlier_report = {}
        
        # Only process numeric columns that are features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_numeric = [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]
        
        for col in feature_numeric:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
            
            # Count outliers
            outliers_low = (data[col] < lower_bound).sum()
            outliers_high = (data[col] > upper_bound).sum()
            total_outliers = outliers_low + outliers_high
            
            if total_outliers > 0:
                outlier_report[col] = {
                    'low': int(outliers_low), 
                    'high': int(outliers_high),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
                # Cap values (winsorization)
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        self.preprocessing_report['outliers_capped'] = outlier_report
        print(f"Capped outliers in {len(outlier_report)} columns")
        return data
    
    def _encode_categoricals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using one-hot encoding."""
        # Find categorical columns that exist in data
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in data.columns]
        
        if cat_cols:
            # For city column (Yishuv), group rare categories to reduce dimensionality
            if 'Yishuv' in cat_cols:
                data = self._group_rare_categories(data, 'Yishuv', threshold=50)
            
            data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
            print(f"Encoded {len(cat_cols)} categorical columns")
        
        return data
    
    def _group_rare_categories(self, data: pd.DataFrame, col: str, threshold: int = 50) -> pd.DataFrame:
        """Group rare categories into 'Other' to reduce dimensionality."""
        value_counts = data[col].value_counts()
        rare_categories = value_counts[value_counts < threshold].index.tolist()
        
        if rare_categories:
            data[col] = data[col].replace(rare_categories, 'Other')
            print(f"  Grouped {len(rare_categories)} rare categories in '{col}' to 'Other'")
        
        return data
    
    def get_preprocessing_report(self) -> dict:
        """Return the preprocessing report."""
        return self.preprocessing_report
    
    def get_feature_names(self) -> List[str]:
        """Return the list of feature names after preprocessing."""
        return self.feature_names
    
    def get_original_ids(self) -> pd.DataFrame:
        """Return the original IDs for output mapping."""
        return self.original_ids


class SyntheticDataLoader(BaseDataLoader):
    def load(self):
        """Loads the CSV and renames columns from Hebrew to English."""
        self.df = pd.read_csv(self.filepath)
        return self.df


class RealDataLoader(BaseDataLoader):
    def load(self):
        """
        Loads the real data. 
        Assumes the data might already have English headers or needs a different mapping.
        For now, we assume it matches the internal structure or we just read it as is.
        You can customize this method for the specific format of the real data.
        """
        self.df = pd.read_csv(self.filepath)
        # Add specific real-data cleaning or mapping here if needed
        return self.df
