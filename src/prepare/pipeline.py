"""
Main orchestration pipeline for NHANES data cleaning, imputation, and encoding.

Coordinates all preprocessing steps using modular components.

Full Pipeline Flow:
1. remove_glucose_nan() - Remove rows with missing glucose
2. bpxdia_nan() - Fix invalid BP measurements
3. remove_id_columns() - Remove ID columns
4. define_feature_types() - Convert to correct data types
5. reconstruct_bmi() - Calculate BMI from weight/height
6. impute_numeric_knn() - KNN imputation for anthropometric vars
7. impute_numeric_median() - Median imputation for other numeric vars
8. impute_categorical_mode() - Mode imputation for categorical vars
9. apply_clinical_rules() - Apply domain knowledge rules
10. encode_categorical() - Encode categorical variables (One-Hot, Ordinal, Target)
"""

import pandas as pd
from src.utils.logger import get_logger

# Import preprocessing modules
from .setup import remove_glucose_nan, bpxdia_nan, remove_id_columns
from .define_types import define_feature_types
from .imputation import (
    impute_numeric_knn,
    impute_numeric_median,
    impute_categorical_mode,
    check_missing_values
)
from .encoding import encode_categorical

logger = get_logger("pipeline")


def reconstruct_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct missing BMI values from weight and height.
    
    BMI = weight_kg / (height_cm / 100)^2
    
    Args:
        df: Input dataframe
    
    Returns:
        DataFrame with reconstructed BMI values
    """
    logger.info("\n[Step 5/10] Reconstructing BMI from weight and height")
    logger.info("-" * 80)
    
    if 'bmi' in df.columns and 'weight_kg' in df.columns and 'height_cm' in df.columns:
        mask_missing_bmi = (
            df['bmi'].isna() &
            df['weight_kg'].notna() &
            df['height_cm'].notna()
        )
        bmi_reconstructed = mask_missing_bmi.sum()
        
        if bmi_reconstructed > 0:
            df.loc[mask_missing_bmi, 'bmi'] = (
                df.loc[mask_missing_bmi, 'weight_kg'] /
                (df.loc[mask_missing_bmi, 'height_cm'] / 100) ** 2
            )
            logger.info(f"✓ Reconstructed {bmi_reconstructed:,} BMI values")
        else:
            logger.info(f"✓ No BMI reconstruction needed")
    else:
        logger.warning("⚠ Required columns (bmi, weight_kg, height_cm) not found")
    
    return df


def apply_clinical_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain knowledge rules based on clinical criteria.
    
    Rules:
    1. If SBP > 140 OR DBP > 90 → hypertension = 1
    2. If age > 60 and hypertension is NaN → hypertension = 1
    
    Args:
        df: Input dataframe
    
    Returns:
        DataFrame with applied clinical rules
    """
    logger.info("\n[Step 9/10] Applying clinical rules")
    logger.info("-" * 80)
    
    # Rule 1: High blood pressure → hypertension
    if 'BPX_SYS' in df.columns and 'BPX_DIA' in df.columns and 'hypertension' in df.columns:
        high_bp_mask = (df['BPX_SYS'] > 140) | (df['BPX_DIA'] > 90)
        updated_hypertension = (high_bp_mask & df['hypertension'].notna()).sum()
        
        if updated_hypertension > 0:
            df.loc[high_bp_mask, 'hypertension'] = 1
            logger.info(f"✓ Marked {updated_hypertension:,} rows as hypertensive (high BP)")
    
    # Rule 2: Age > 60 → assume hypertension if missing
    if 'age_years' in df.columns and 'hypertension' in df.columns:
        age_rule_mask = df['hypertension'].isna() & (df['age_years'] > 60)
        age_rule_count = age_rule_mask.sum()
        
        if age_rule_count > 0:
            df.loc[age_rule_mask, 'hypertension'] = 1
            logger.info(f"✓ Marked {age_rule_count:,} rows as hypertensive (age > 60)")
    
    logger.info(f"✓ Clinical rules applied")
    
    return df


def clean_and_impute(df: pd.DataFrame,
                     encoding_config: dict = None) -> tuple:
    """
    Full preprocessing pipeline for NHANES health dataset.
    
    Pipeline steps:
    1. [setup.py] Remove rows with NaN glucose_value (destructive filtering)
    2. [setup.py] Replace BPX_DIA = 0 with NaN (value transformation)
    3. [setup.py] Remove ID columns (SEQN, subject_id, ID)
    4. [define_types.py] Define feature types (binary, categorical, int, numeric)
    5. [THIS MODULE] Reconstruct BMI from weight and height
    6. [imputation.py] Numeric imputation (KNN for anthropometric, median for others)
    7. [imputation.py] Categorical imputation (mode)
    8. [THIS MODULE] Apply clinical rules
    9. [imputation.py] Check missing values
    10. [encoding.py] Encode categorical variables (One-Hot, Ordinal, Target)
    
    Args:
        df: Raw input dataframe
        encoding_config: Dictionary with encoding configuration
        
        Example: {
            'onehot_cols': ['gender', 'ethnicity'],
            'ordinal_cols': [],
            'target_cols': {
                'columns': ['smoker'],
                'target': 'diabetes_risk',
                'smoothing': 1.0
            }
        }
    
    Returns:
        Tuple of (encoded_df, encoders_dict)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FULL PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    # ========================================================================
    # Step 1: Remove rows with NaN glucose_value (destructive)
    # ========================================================================
    logger.info("\n[Step 1/10] Filtering rows with NaN glucose_value")
    logger.info("-" * 80)
    df = remove_glucose_nan(df)
    logger.info(f"Rows after glucose filtering: {len(df):,}")
    
    # ========================================================================
    # Step 2: Replace invalid BPX_DIA values with NaN (transformation)
    # ========================================================================
    logger.info("\n[Step 2/10] Replacing invalid BPX_DIA = 0 with NaN")
    logger.info("-" * 80)
    df = bpxdia_nan(df)
    logger.info(f"BPX_DIA transformation completed")
    
    # ========================================================================
    # Step 3: Remove ID columns
    # ========================================================================
    logger.info("\n[Step 3/10] Removing ID columns")
    logger.info("-" * 80)
    df = remove_id_columns(df, id_cols=['SEQN', 'subject_id', 'ID'])
    
    # ========================================================================
    # Step 4: Define feature types
    # ========================================================================
    logger.info("\n[Step 4/10] Defining feature types")
    logger.info("-" * 80)
    df = define_feature_types(df)
    logger.info(f"✓ Feature types defined successfully")
    
    # ========================================================================
    # Step 5: BMI reconstruction
    # ========================================================================
    df = reconstruct_bmi(df)
    
    # ========================================================================
    # Step 6: Numerical imputation (KNN + Median)
    # ========================================================================
    logger.info("\n[Step 6/10] Numerical imputation")
    logger.info("-" * 80)
    
    # KNN for correlated anthropometric variables
    knn_cols = ["waist_cm", "weight_kg", "height_cm", "bmi"]
    df = impute_numeric_knn(df, knn_cols, n_neighbors=5)
    
    # Median for independent numeric variables
    numeric_cols = [
        "creatinine", "bmi", "waist_cm", "weight_kg", "height_cm",
        "BPX_SYS", "BPX_DIA", "glucose_value", "hdl_cholesterol",
        "total_cholesterol", "TRIGLY", "SLQ"
    ]
    df = impute_numeric_median(df, numeric_cols, skip_cols=knn_cols)
    
    # ========================================================================
    # Step 7: Categorical imputation
    # ========================================================================
    logger.info("\n[Step 7/10] Categorical imputation")
    logger.info("-" * 80)
    categorical_cols = [
        "gender", "ethnicity", "smoker", "hypertension",
        "liver_disease", "heart_disease", "income_poverty_ratio"
    ]
    df = impute_categorical_mode(df, categorical_cols)
    
    # ========================================================================
    # Step 8: Clinical rules
    # ========================================================================
    df = apply_clinical_rules(df)
    
    # ========================================================================
    # Step 9: Check missing values before encoding
    # ========================================================================
    logger.info("\n[Step 8/10] Pre-encoding validation")
    logger.info("-" * 80)
    missing_stats = check_missing_values(df, verbose=True)
    
    if missing_stats['total_missing'] > 0:
        logger.warning(f"\n⚠ {missing_stats['total_missing']:,} missing values remain before encoding")
    
    # ========================================================================
    # Step 10: Categorical encoding
    # ========================================================================
    logger.info("\n[Step 10/10] Categorical encoding")
    logger.info("-" * 80)
    
    encoders = {}
    if encoding_config:
        df, encoders = encode_categorical(
            df,
            onehot_cols=encoding_config.get('onehot_cols', []),
            ordinal_cols=encoding_config.get('ordinal_cols', []),
            target_cols=encoding_config.get('target_cols', None)
        )
    else:
        logger.info("⊘ No encoding configuration provided")
        logger.info("  Skipping categorical encoding")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    logger.info(f"Initial shape: {initial_rows:,} rows × {initial_cols} columns")
    logger.info(f"Final shape: {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Rows removed: {initial_rows - len(df):,} ({((initial_rows - len(df)) / initial_rows * 100):.2f}%)")
    logger.info(f"New columns created by encoding: {len(df.columns) - initial_cols}")
    
    # Final check for missing values
    final_missing = df.isnull().sum().sum()
    
    if final_missing == 0:
        logger.info(f"\n✓ ✓ ✓ ZERO MISSING VALUES IN FINAL DATASET ✓ ✓ ✓")
    else:
        logger.warning(f"\n⚠ {final_missing:,} missing values in final dataset")
    
    logger.info("=" * 80)
    
    return df, encoders


if __name__ == "__main__":
    logger.info("Loading dataset from Parquet")
    df = pd.read_parquet("./data/nhanes_data/NHANES_consolidated.parquet")
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Define encoding configuration
    encoding_config = {
        'onehot_cols': ['gender', 'ethnicity'],
        'ordinal_cols': [],
        'target_cols': None  # Set to None to skip target encoding
    }
    
    logger.info("Running full preprocessing pipeline")
    df_processed, encoders = clean_and_impute(df, encoding_config=encoding_config)
    
    logger.info("Saving processed dataset to Parquet")
    df_processed.to_parquet("./data/nhanes_data/NHANES_final.parquet", index=False)
    logger.info("✓ NHANES_final.parquet saved successfully")
