from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any
from config import COLUMN_MAPPING, TARGET_COL

class BaseDataLoader(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

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

class SyntheticDataLoader(BaseDataLoader):
    def load(self):
        """Loads the CSV and renames columns from Hebrew to English."""
        self.df = pd.read_csv(self.filepath)
        
        # Rename columns
        # Only rename columns that exist in the mapping
        rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in self.df.columns}
        self.df.rename(columns=rename_dict, inplace=True)
        
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
