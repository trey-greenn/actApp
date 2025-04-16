import pandas as pd
from typing import Dict

REQUIRED_COLUMNS = [
    "transaction_id", 
    "amount", 
    "origin_country", 
    "destination_country", 
    "timestamp", 
    "account_id", 
    "label"
]

def validate_csv_data(df: pd.DataFrame) -> Dict:
    """
    Validate that the CSV data contains the required columns and the correct data types
    """
    # Check if all required columns are present
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        return {
            "is_valid": False,
            "error": f"Missing required columns: {', '.join(missing_columns)}"
        }
    
    # Check for null values in required columns
    null_columns = [col for col in REQUIRED_COLUMNS if df[col].isnull().any()]
    if null_columns:
        return {
            "is_valid": False,
            "error": f"Null values found in columns: {', '.join(null_columns)}"
        }
    
    # Check if label column contains only 0 and 1
    if not all(df['label'].isin([0, 1])):
        return {
            "is_valid": False,
            "error": "Label column must contain only 0 (not suspicious) or 1 (suspicious)"
        }
    
    # Check if amount is numeric
    if not pd.api.types.is_numeric_dtype(df['amount']):
        return {
            "is_valid": False,
            "error": "Amount column must be numeric"
        }
    
    # Check if there are duplicate transaction_ids
    if df['transaction_id'].duplicated().any():
        return {
            "is_valid": False,
            "error": "Duplicate transaction_id values found"
        }
    
    # Validation successful
    return {
        "is_valid": True
    }