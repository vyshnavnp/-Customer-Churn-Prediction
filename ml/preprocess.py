import pandas as pd
import numpy as np
from typing import Tuple, List

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from Excel or CSV.
    """
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning based on EDA logic:
    1. Drops ID columns.
    2. Converts TotalCharges to numeric (coercing errors).
    3. Fills missing numerical values with median.
    4. Maps Churn to binary 0/1.
    """
    df = df.copy()

    # 1. Drop irrelevant columns
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 2. Handle TotalCharges (often read as object due to empty strings)
    # Coerce errors will turn empty strings " " into NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 3. Handle Missing Values
    # Fill missing TotalCharges with median (robust to outliers)
    if df['TotalCharges'].isnull().sum() > 0:
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)

    # 4. Encode Target
    # Ensure Churn is binary 0/1. If it's already 0/1, map won't break it
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies categorical and numerical features for CatBoost.
    Returns: (categorical_features_list, numerical_features_list)
    """
    # Exclude target
    features = df.drop(columns=['Churn'])
    
    # Identify based on dtypes
    # CatBoost handles object/category/string as categorical
    cat_features = features.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Integers and Floats are numerical
    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return cat_features, num_features

def prepare_for_training(file_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Orchestrator function to load, clean, and split X, y.
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        cat_features (List[str]): List of categorical column names for CatBoost
    """
    # Load
    raw_df = load_data(file_path)
    
    # Clean
    cleaned_df = clean_data(raw_df)
    
    # Split Features/Target
    X = cleaned_df.drop(columns=['Churn'])
    y = cleaned_df['Churn']
    
    # Identify Categorical Columns for CatBoost
    cat_features, _ = get_feature_lists(cleaned_df)
    
    # Ensure categorical columns are strictly strings (CatBoost requirement)
    for col in cat_features:
        X[col] = X[col].astype(str)
        
    return X, y, cat_features

if __name__ == "__main__":
    # Test run
    try:
        # Assuming you have the data in a folder named 'data' one level up or same level
        # Adjust path as necessary for your local test
        sample_path = "data/Customer_data.csv" 
        X, y, cats = prepare_for_training(sample_path)
        
        print("Data Preprocessing Successful!")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Categorical features detected: {cats}")
        print("\nSample Data:")
        print(X.head())
    except Exception as e:
        print(f"Error during test run: {e}")