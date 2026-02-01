"""
Imputation strategies for numeric and categorical variables.
Separates KNN, median, and mode imputation logic into reusable functions.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.utils.logger import get_logger

logger = get_logger("preprocessing.imputation")


# =============================================================================
# KNN IMPUTATION
# =============================================================================
def impute_numeric_knn(
    df: pd.DataFrame,
    cols: list,
    n_neighbors: int,
    mode: str,
    artifacts: dict
) -> pd.DataFrame:

    """
    Apply KNN imputation to numeric columns.

    Args:
        df: Input dataframe
        knn_cols: List of column names to impute with KNN
        n_neighbors: Number of neighbors for KNN (default: 5)
        
    Returns:
        DataFrame with KNN imputed values
    """
    cols = [c for c in cols if c in df.columns]

    if not cols:
        return df

    if mode == "fit":
        logger.info("Applying KNN imputation (FIT)")
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[cols] = imputer.fit_transform(df[cols])

        artifacts.setdefault("imputation", {})
        artifacts["imputation"]["knn"] = imputer

    else:
        logger.info("Applying KNN imputation (TRANSFORM)")
        imputer = artifacts["imputation"]["knn"]
        df[cols] = imputer.transform(df[cols])

    return df


# =============================================================================
# MEDIAN IMPUTATION
# =============================================================================
def impute_numeric_median(
    df: pd.DataFrame,
    cols: list,
    mode: str,
    artifacts: dict
) -> pd.DataFrame:
    """
    Apply median imputation to numeric columns.
    
    Args:
        df: Input dataframe
        numeric_cols: List of all numeric columns to process
        skip_cols: Columns to skip (already imputed with KNN, etc.)
        
    Returns:
        DataFrame with median imputed values
    """
    df = df.copy()
    cols = [c for c in cols if c in df.columns]

    if not cols:
        return df

    if mode == "fit":
        logger.info("Applying median imputation (FIT)")
        medians = {}

        for col in cols:
            median = df[col].median()
            medians[col] = median
            missing = df[col].isna().sum()
            if missing > 0:
                logger.info(f"  {col}: {missing} values imputed with median ({median:.2f})")
                df[col] = df[col].fillna(median)

        if pd.notna(median):
            df[col] = df[col].fillna(median)
            logger.info(f"{col} imputed with median ({median:.2f})")
        logger.info("\n")

        artifacts.setdefault("imputation", {})
        artifacts["imputation"]["median"] = medians

    else:
        logger.info("Applying median imputation (TRANSFORM)")
        medians = artifacts["imputation"]["median"]

        for col, median in medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        
        if pd.notna(median):
            df[col] = df[col].fillna(median)
            logger.info(f"{col} imputed with median ({median:.2f})")
        logger.info("\n")
    return df


# =============================================================================
# MODE IMPUTATION
# =============================================================================
def impute_categorical_mode(
    df: pd.DataFrame,
    cols: list,
    mode: str,
    artifacts: dict
) -> pd.DataFrame:
    """
    Apply mode (most frequent value) imputation to categorical columns.
    
    Best for: Categorical variables with clear majority class
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical columns to impute
        
    Returns:
        DataFrame with mode imputed values
    """
    df = df.copy()
    cols = [c for c in cols if c in df.columns]

    if not cols:
        return df

    if mode == "fit":
        logger.info("Applying mode imputation (FIT)")
        modes = {}

        for col in cols:
            if df[col].isna().any():
                mode_val = df[col].mode(dropna=True)[0]
                modes[col] = mode_val
                missing = df[col].isna().sum()
                logger.info(f"{col}: {missing} values imputed with mode ({mode_val})")
                df[col] = df[col].fillna(mode_val)

        artifacts.setdefault("imputation", {})
        artifacts["imputation"]["mode"] = modes

    else:
        logger.info("Applying mode imputation (TRANSFORM)")
        modes = artifacts["imputation"]["mode"]

        for col, mode_val in modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

    return df


# =============================================================================
# MISSING VALUES
# =============================================================================
def check_missing_values(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check and report missing values in dataframe.
    
    Args:
        df: Input dataframe
        verbose: Print detailed report
        
    Returns:
        Dictionary with missing value statistics
    """
    logger.info("Checking missing values")
    
    nan_summary = df.isnull().sum()
    total_missing = nan_summary.sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells) * 100
    
    if verbose:
        logger.info(f"Total missing: {total_missing:,} ({missing_pct:.2f}%)\n")
    
    if total_missing > 0:
        cols_with_missing = nan_summary[nan_summary > 0].sort_values(ascending=False)
        if verbose:
            logger.info(f"Columns with missing values ({len(cols_with_missing)}):\n")
            for col, count in cols_with_missing.items():
                pct = (count / len(df)) * 100
                logger.info(f"{col}: {count:,} ({pct:.2f}%)\n")
    else:
        logger.info("No missing values found\n")