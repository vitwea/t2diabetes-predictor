"""
Imputation strategies for numeric and categorical variables.
Separates KNN, median, and mode imputation logic into reusable functions.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.utils.logger import get_logger

logger = get_logger("imputation")


def impute_numeric_knn(df: pd.DataFrame, knn_cols: list, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Apply KNN imputation to numeric columns.
    
    Best for: Correlated numeric variables (anthropometric measures)
    - weight_kg, height_cm, waist_cm, bmi
    
    Args:
        df: Input dataframe
        knn_cols: List of column names to impute with KNN
        n_neighbors: Number of neighbors for KNN (default: 5)
        
    Returns:
        DataFrame with KNN imputed values
    """
    logger.info("Applying KNN imputation")
    logger.info(f"  Columns: {knn_cols}")
    logger.info(f"  n_neighbors: {n_neighbors}")
    
    # Filter to columns that exist in dataframe
    knn_cols = [col for col in knn_cols if col in df.columns]
    
    if not knn_cols:
        logger.warning("⚠ No valid columns found for KNN imputation")
        return df
    
    # Count missing before
    missing_before = df[knn_cols].isnull().sum().sum()
    
    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    df[knn_cols] = knn_imputer.fit_transform(df[knn_cols])
    
    # Count missing after
    missing_after = df[knn_cols].isnull().sum().sum()
    
    logger.info(f"✓ KNN imputation completed")
    logger.info(f"  Missing before: {missing_before:,}")
    logger.info(f"  Missing after: {missing_after:,}")
    logger.info(f"  Imputed: {missing_before - missing_after:,} values")
    
    return df


def impute_numeric_median(df: pd.DataFrame, numeric_cols: list, skip_cols: list = None) -> pd.DataFrame:
    """
    Apply median imputation to numeric columns.
    
    Best for: Independent numeric variables
    - Blood pressure, glucose, cholesterol, triglycerides, creatinine
    
    Args:
        df: Input dataframe
        numeric_cols: List of all numeric columns to process
        skip_cols: Columns to skip (already imputed with KNN, etc.)
        
    Returns:
        DataFrame with median imputed values
    """
    logger.info("Applying median imputation")
    
    if skip_cols is None:
        skip_cols = []
    
    # Filter columns to impute
    cols_to_impute = [col for col in numeric_cols if col in df.columns and col not in skip_cols]
    
    if not cols_to_impute:
        logger.warning("⚠ No columns to impute with median")
        return df
    
    logger.info(f"  Columns: {cols_to_impute}")
    
    for col in cols_to_impute:
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            logger.info(f"  {col}: No missing values")
            continue
        
        median_val = df[col].median()
        
        if pd.notna(median_val):
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: {missing_count:,} values imputed with median ({median_val:.2f})")
        else:
            # Fallback: fill with 0 if median is NaN (column mostly empty)
            df[col] = df[col].fillna(0)
            logger.warning(f"  {col}: {missing_count:,} values imputed with 0 (column was mostly NaN)")
    
    logger.info(f"✓ Median imputation completed")
    
    return df


def impute_numeric_mean(df: pd.DataFrame, numeric_cols: list, skip_cols: list = None) -> pd.DataFrame:
    """
    Apply mean imputation to numeric columns (alternative to median).
    
    Note: Median is usually preferred (more robust to outliers)
    
    Args:
        df: Input dataframe
        numeric_cols: List of all numeric columns to process
        skip_cols: Columns to skip
        
    Returns:
        DataFrame with mean imputed values
    """
    logger.info("Applying mean imputation")
    
    if skip_cols is None:
        skip_cols = []
    
    cols_to_impute = [col for col in numeric_cols if col in df.columns and col not in skip_cols]
    
    if not cols_to_impute:
        logger.warning("⚠ No columns to impute with mean")
        return df
    
    logger.info(f"  Columns: {cols_to_impute}")
    
    for col in cols_to_impute:
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            logger.info(f"  {col}: No missing values")
            continue
        
        mean_val = df[col].mean()
        
        if pd.notna(mean_val):
            df[col] = df[col].fillna(mean_val)
            logger.info(f"  {col}: {missing_count:,} values imputed with mean ({mean_val:.2f})")
        else:
            df[col] = df[col].fillna(0)
            logger.warning(f"  {col}: {missing_count:,} values imputed with 0")
    
    logger.info(f"✓ Mean imputation completed")
    
    return df


def impute_categorical_mode(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Apply mode (most frequent value) imputation to categorical columns.
    
    Best for: Categorical variables with clear majority class
    - gender, ethnicity, smoker, hypertension, liver_disease, heart_disease, diabetes_risk
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical columns to impute
        
    Returns:
        DataFrame with mode imputed values
    """
    logger.info("Applying mode imputation")
    logger.info(f"  Columns: {categorical_cols}")
    
    for col in categorical_cols:
        if col not in df.columns:
            logger.warning(f"  {col}: Column not found")
            continue
        
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            logger.info(f"  {col}: No missing values")
            continue
        
        mode_val = df[col].mode()
        
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
            logger.info(f"  {col}: {missing_count:,} values imputed with mode ({mode_val[0]})")
        else:
            logger.warning(f"  {col}: Could not compute mode (column might be empty)")
    
    logger.info(f"✓ Mode imputation completed")
    
    return df


def impute_categorical_forward_fill(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Apply forward fill imputation to categorical columns.
    
    Note: Only suitable for time-series data where forward fill makes sense.
    Usually NOT recommended for cross-sectional health data.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical columns to impute
        
    Returns:
        DataFrame with forward filled values
    """
    logger.info("Applying forward fill imputation")
    logger.info(f"  Columns: {categorical_cols}")
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            continue
        
        df[col] = df[col].fillna(method='ffill')
        
        # Backward fill for any remaining NaN (start of series)
        df[col] = df[col].fillna(method='bfill')
        
        new_missing = df[col].isnull().sum()
        logger.info(f"  {col}: {missing_count:,} values filled, {new_missing:,} remain")
    
    logger.info(f"✓ Forward fill imputation completed")
    
    return df


def check_missing_values(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check and report missing values in dataframe.
    
    Args:
        df: Input dataframe
        verbose: Print detailed report
        
    Returns:
        Dictionary with missing value statistics
    """
    logger.info("\nChecking missing values...")
    
    nan_summary = df.isnull().sum()
    total_missing = nan_summary.sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells) * 100
    
    if verbose:
        logger.info(f"Total missing: {total_missing:,} ({missing_pct:.2f}%)")
    
    if total_missing > 0:
        cols_with_missing = nan_summary[nan_summary > 0].sort_values(ascending=False)
        if verbose:
            logger.info(f"Columns with missing values ({len(cols_with_missing)}):")
            for col, count in cols_with_missing.items():
                pct = (count / len(df)) * 100
                logger.info(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("✓ No missing values found")
    
    return {
        'total_missing': total_missing,
        'total_cells': total_cells,
        'missing_pct': missing_pct,
        'cols_with_missing': nan_summary[nan_summary > 0].to_dict()
    }
