"""
Feature Type Definition and Transformation
Define data types for each variable and apply transformations
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("define_types")


def define_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define and transform feature types according to specifications:
    
    BINARY (0/1):
    - smoker: 1=Yes -> 1, rest -> 0
    - liver_disease: 1=Yes -> 1, rest -> 0
    - heart_disease: 1=Yes -> 1, rest -> 0
    - hypertension: 1=Yes -> 1, rest -> 0
    
    INT:
    - age_years
    
    CATEGORY:
    - ethnicity
    - income_poverty_ratio
    
    NUMERIC (float, as is):
    - All other features remain numeric
    """
    
    logger.info("=" * 80)
    logger.info("DEFINE FEATURE TYPES AND TRANSFORMATIONS")
    logger.info("=" * 80)
    
    df = df.copy()
    
    # ========================================================================
    # BINARY VARIABLES: 1=Yes, rest=No (inverse: 0=Yes, 1=No)
    # ========================================================================
    
    binary_vars = ["smoker", "liver_disease", "heart_disease", "hypertension"]
    
    logger.info("\n" + "-" * 80)
    logger.info("BINARY VARIABLES (1=Yes, rest=No → 0=Yes, 1=No)")
    logger.info("-" * 80)
    
    for col in binary_vars:
        if col in df.columns:
            before = df[col].value_counts().to_dict()
            logger.info(f"\n{col}:")
            logger.info(f"  Before transformation: {before}")
            
            # Transform: 1=Yes -> 1, rest -> 0
            df[col] = ((df[col] == 1) | (df[col] == 1.0)).astype(int)
            
            # Convert to int64
            df[col] = df[col].astype("int64")
            
            after = df[col].value_counts().to_dict()
            logger.info(f"  After transformation: {after}")
            logger.info(f"  Data type: {df[col].dtype}")
        else:
            logger.warning(f"  Column '{col}' not found in dataframe")
    
    # ========================================================================
    # INTEGER VARIABLES
    # ========================================================================
    
    int_vars = ["age_years"]
    
    logger.info("\n" + "-" * 80)
    logger.info("INTEGER VARIABLES")
    logger.info("-" * 80)
    
    for col in int_vars:
        if col in df.columns:
            logger.info(f"\n{col}:")
            logger.info(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Mean: {df[col].mean():.2f}")
            
            # Convert to int64
            df[col] = df[col].astype("int64")
            
            logger.info(f"  Data type: {df[col].dtype}")
        else:
            logger.warning(f"  Column '{col}' not found in dataframe")
    
    # ========================================================================
    # CATEGORICAL VARIABLES
    # ========================================================================
    
    cat_vars = ["ethnicity", "income_poverty_ratio"]
    
    logger.info("\n" + "-" * 80)
    logger.info("CATEGORICAL VARIABLES")
    logger.info("-" * 80)
    
    for col in cat_vars:
        if col in df.columns:
            logger.info(f"\n{col}:")
            unique_count = df[col].nunique()
            logger.info(f"  Unique values: {unique_count}")
            
            # Show value counts
            val_counts = df[col].value_counts().head(10)
            for val, count in val_counts.items():
                pct = (count / len(df)) * 100
                logger.info(f"    {val}: {count:,} ({pct:.2f}%)")
            
            # Convert to category
            df[col] = df[col].astype("category")
            
            logger.info(f"  Data type: {df[col].dtype}")
        else:
            logger.warning(f"  Column '{col}' not found in dataframe")
    
    # ========================================================================
    # REMAINING NUMERIC VARIABLES (float64, no transformation)
    # ========================================================================
    
    logger.info("\n" + "-" * 80)
    logger.info("NUMERIC VARIABLES (unchanged)")
    logger.info("-" * 80)
    
    # Get all columns not yet processed
    processed_cols = (
        ["diabetes_risk", "subject_id"] +  # Target and ID
        binary_vars +
        int_vars +
        cat_vars
    )
    
    numeric_cols = [col for col in df.columns if col not in processed_cols]
    
    if numeric_cols:
        logger.info(f"\nNumeric columns (remaining as float64):")
        for col in numeric_cols:
            logger.info(f"  {col}: {df[col].dtype}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL DATA TYPES")
    logger.info("=" * 80)
    
    dtype_summary = df.dtypes.value_counts()
    logger.info(f"\nData type distribution:")
    for dtype, count in dtype_summary.items():
        logger.info(f"  {dtype}: {count} columns")
    
    logger.info(f"\nDetailed column types:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].dtype}")
    
    return df
