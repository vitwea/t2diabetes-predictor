"""
Remove rows with NaN in glucose_value
Replace diastolic_bp = 0 with NaN (invalid measurement)
Keep only rows with actual glucose measurements
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("preprocessing.cleaning")


def remove_glucose_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows where glucose_value is NaN.
    Replace diastolic_bp = 0 with NaN (invalid blood pressure measurement).
    Keep only rows with actual glucose measurements.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with rows containing NaN in glucose_value removed and diastolic_bp=0 replaced with NaN
    """
    
    logger.info("=" * 80)
    logger.info("DATA CLEANING: Remove NaN glucose_value")
    logger.info("=" * 80 + "\n")
    
    initial_rows = len(df)
    
    # ========================================================================
    # Step 1: Remove rows with NaN in glucose_value
    # ========================================================================
    
    logger.info("[Step 1] Removing rows with NaN in glucose_value and deleting variable")
    logger.info("-" * 80)
    
    # Check if column exists
    if 'glucose_value' not in df.columns:
        logger.warning("Column 'glucose_value' not found in dataframe")
        return df
    
    # Count NaN values
    nan_count = df['glucose_value'].isnull().sum()
    nan_pct = (nan_count / len(df)) * 100
    
    logger.info(f"Initial rows: {initial_rows:,}")
    logger.info(f"Rows with NaN in glucose_value: {nan_count:,} ({nan_pct:.2f}%)")
    logger.info(f"Rows with valid glucose_value: {initial_rows - nan_count:,}")
    
    # Remove rows with NaN in glucose_value
    df = df[df['glucose_value'].notna()]

    # Delete 'glucose_value column'
    df = df.drop(columns=['glucose_value'])
    
    removed_rows = initial_rows - len(df)


    if 'glucose_value' not in df.columns:
        logger.info("Column 'glucose_value' dleted from the dataframe")
    logger.info(f"Removed {removed_rows:,} rows with NaN glucose_value")
    logger.info(f"Rows retained after filter: {len(df):,} ({(len(df) / initial_rows * 100):.2f}%)\n")

    return df
    
def bpxdia_nan(df: pd.DataFrame) -> pd.DataFrame:
    # ========================================================================
    # Step 2: Replace diastolic_bp = 0 with NaN
    # ========================================================================

    logger.info("[Step 2] Replacing diastolic_bp = 0 with NaN")
    logger.info("-" * 80)
    
    if 'diastolic_bp' in df.columns:
        bpxdia_zeros = (df['diastolic_bp'] == 0).sum()
        
        if bpxdia_zeros > 0:
            logger.info(f"Found {bpxdia_zeros:,} rows with diastolic_bp = 0")
            logger.info(f"  ({(bpxdia_zeros / len(df) * 100):.2f}% of current data)")
            
            # Replace 0 with NaN
            df['diastolic_bp'] = df['diastolic_bp'].replace(0, np.nan)
            
            logger.info(f"Replaced all diastolic_bp = 0 with NaN\n")
        else:
            logger.info(f"No diastolic_bp = 0 values found\n")
    else:
        logger.warning("Column 'diastolic_bp' not found in dataframe\n")
    
    return df

def remove_id_columns(df: pd.DataFrame, id_cols: list = None) -> pd.DataFrame:
    """
    Remove ID columns that are not needed for modeling.
    
    ID columns typically have one unique value per row and don't contribute
    to predictive power. Examples: subject_id, participant_id, etc.
    
    Args:
        df: Input dataframe
        id_cols: List of column names to remove (default: ['subject_id'])
        
    Returns:
        DataFrame with ID columns removed
    """
    logger.info("[Step 3] Removing ID columns")
    logger.info("-" * 80)
    
    if id_cols is None:
        id_cols = ['subject_id']
    
    cols_to_remove = [col for col in id_cols if col in df.columns]
    
    if not cols_to_remove:
        logger.info("No ID columns found to remove")
        return df
    
    logger.info(f"Columns to remove: {cols_to_remove}")
    
    for col in cols_to_remove:
        df = df.drop(columns=[col])
        logger.info(f"Removed: {col}")
    
    logger.info(f"ID columns removed successfully\n")
    
    return df