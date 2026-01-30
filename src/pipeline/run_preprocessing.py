"""
UPDATED run_preprocessing.py
Integración con feature_selection.py

Pipeline:
1. [cleaning] Remove rows with missing glucose
2. [cleaning] Fix invalid BP measurements
3. [cleaning] Remove ID columns
4. [types] Define feature types
5. [clinical_rules] Reconstruct BMI
6. [imputation] KNN imputation for anthropometric
7. [imputation] Median imputation for numeric
8. [imputation] Mode imputation for categorical
9. [clinical_rules] Apply domain knowledge rules
10. [feature_engineering] Create engineered features
11. [encoding] Encode categorical variables
12. [feature_selection] SELECT TOP 20 FEATURES ← Imported from feature_selection.py
"""

import pandas as pd
import pickle
import logging

# Import preprocessing modules
from src.preprocessing.cleaning import remove_glucose_nan, bpxdia_nan, remove_id_columns
from src.features.clinical_rules import reconstruct_bmi, apply_clinical_rules
from src.preprocessing.define_types import define_feature_types
from src.preprocessing.imputation import (impute_numeric_knn,impute_numeric_median,impute_categorical_mode,check_missing_values)
from src.preprocessing.encoding import encode_categorical
from src.features.feature_engineering import create_all_features
from src.features.feature_selection import select_top_features, save_features

# Setup logger
logger = logging.getLogger("preprocessing")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSING FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def clean_and_impute(df: pd.DataFrame,
                     encoding_config: dict = None,
                     select_top20: bool = True) -> tuple:
    """
    Full preprocessing pipeline with optional TOP 20 feature selection.
    
    Pipeline steps:
    1. Remove rows with NaN glucose_value
    2. Replace invalid diastolic_bp values
    3. Remove ID columns
    4. Define feature types
    5. Reconstruct BMI
    6. Numeric imputation (KNN + Median)
    7. Categorical imputation (Mode)
    8. Apply clinical rules
    9. Create feature engineering features
    10. Check missing values
    11. Categorical encoding
    12. SELECT TOP 20 FEATURES (if select_top20=True) ← FROM feature_selection.py
    
    Args:
        df: Raw input dataframe
        encoding_config: Dictionary with encoding configuration
        select_top20: Whether to select TOP 20 features (default: True)
    
    Returns:
        Tuple of (processed_df, encoders_dict)
    
    Example:
        encoding_config = {
            'ordinal_cols': ['bmi_category', 'bp_category', 'age_group', 'ethnicity'],
            'onehot_cols': [],
            'target_cols': None
        }
        
        df_processed, encoders = clean_and_impute(
            df, 
            encoding_config=encoding_config,
            select_top20=True
        )
    """
    
    logger.info("\n" + "="*80)
    logger.info("STARTING FULL PREPROCESSING PIPELINE")
    logger.info("="*80)
    
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 1: Remove rows with NaN glucose_value
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 1/12] Filtering rows with NaN glucose_value")
    logger.info("-"*80)
    df = remove_glucose_nan(df)
    logger.info(f"Rows after glucose filtering: {len(df):,}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 2: Replace invalid diastolic_bp values
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 2/12] Replacing invalid diastolic_bp = 0 with NaN")
    logger.info("-"*80)
    df = bpxdia_nan(df)
    logger.info(f"diastolic_bp transformation completed")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 3: Remove ID columns
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 3/12] Removing ID columns")
    logger.info("-"*80)
    df = remove_id_columns(df, id_cols=['SEQN', 'subject_id', 'ID'])
    logger.info(f"ID columns removed")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 4: Define feature types
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 4/12] Defining feature types")
    logger.info("-"*80)
    df = define_feature_types(df)
    logger.info(f"Feature types defined")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 5: BMI reconstruction
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 5/12] Reconstructing BMI")
    logger.info("-"*80)
    df = reconstruct_bmi(df)
    logger.info(f"BMI reconstructed")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 6: Numerical imputation
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 6/12] Numerical imputation")
    logger.info("-"*80)
    
    knn_cols = ["waist_cm", "weight_kg", "height_cm", "bmi"]
    df = impute_numeric_knn(df, knn_cols, n_neighbors=5)
    
    numeric_cols = [
        "creatinine", "bmi", "waist_cm", "weight_kg", "height_cm",
        "systolic_bp", "diastolic_bp", "glucose_value", "hdl_cholesterol",
        "total_cholesterol", "triglycerides", "sleep_hours"
    ]
    df = impute_numeric_median(df, numeric_cols, skip_cols=knn_cols)
    logger.info(f"Numerical imputation completed")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 7: Categorical imputation
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 7/12] Categorical imputation")
    logger.info("-"*80)
    
    categorical_cols = [
        "gender", "ethnicity", "smoker", "hypertension",
        "liver_disease", "heart_disease", "income_poverty_ratio"
    ]
    df = impute_categorical_mode(df, categorical_cols)
    logger.info(f"Categorical imputation completed")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 8: Clinical rules
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 8/12] Applying clinical rules")
    logger.info("-"*80)
    df = apply_clinical_rules(df)
    logger.info(f"Clinical rules applied")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 9: Feature Engineering
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 9/12] Creating engineered features")
    logger.info("-"*80)
    df = create_all_features(df)
    logger.info(f"Feature engineering completed")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 10: Check missing values
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 10/12] Pre-encoding validation")
    logger.info("-"*80)
    missing_stats = check_missing_values(df, verbose=True)
    
    if missing_stats['total_missing'] > 0:
        logger.warning(f"{missing_stats['total_missing']:,} missing values remain")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 11: Categorical encoding
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n[Step 11/12] Categorical encoding")
    logger.info("-"*80)
    
    encoders = {}
    if encoding_config:
        df, encoders = encode_categorical(
            df,
            onehot_cols=encoding_config.get('onehot_cols', []),
            ordinal_cols=encoding_config.get('ordinal_cols', []),
            target_cols=encoding_config.get('target_cols', None)
        )
        logger.info(f"Categorical encoding completed")
    else:
        logger.info("No encoding configuration provided")
        logger.info("Skipping categorical encoding")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Step 12: FEATURE SELECTION 
    # ════════════════════════════════════════════════════════════════════════════
    
    if select_top20:
        logger.info("\n[Step 12/12] Selecting TOP features")
        logger.info("-"*80)
        
        df, selected_features = select_top_features(df)
        save_features(selected_features)
        
    else:
        logger.info("\n[Step 12/12] Skipping TOP 20 feature selection")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Final Summary
    # ════════════════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "="*80)
    logger.info("FULL PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    logger.info(f"Initial shape:     {initial_rows:,} rows × {initial_cols} columns")
    logger.info(f"Final shape:       {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Rows removed:      {initial_rows - len(df):,} ({((initial_rows - len(df)) / initial_rows * 100):.2f}%)")
    
    # Final check for missing values
    final_missing = df.isnull().sum().sum()
    if final_missing == 0:
        logger.info(f"ZERO MISSING VALUES IN FINAL DATASET")
    else:
        logger.warning(f"{final_missing:,} missing values in final dataset")
    
    logger.info("="*80)
    
    return df, encoders

# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    logger.info("Loading dataset from Parquet")
    df = pd.read_parquet("data/nhanes_data/cleaned/dataset_cleaned.parquet")
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Define encoding configuration
    encoding_config = {
        'onehot_cols': [],
        'ordinal_cols': ['bmi_category', 'bp_category', 'age_group', 'ethnicity'],
        'target_cols': None
    }
    
    logger.info("Running full preprocessing + feature selection pipeline")
    
    # Run pipeline WITH feature selection
    df_processed, encoders = clean_and_impute(
        df,
        encoding_config=encoding_config,
        select_top20=True  # ← SELECCIONA TOP 20 AUTOMÁTICAMENTE
    )
    
    logger.info("Saving processed dataset to Parquet")
    
    output_path = "data/dataset/dataset_final.parquet"
    df_processed.to_parquet(output_path, index=False)
    logger.info(f"Dataset saved: {output_path}")
    
    if encoders:
        encoders_path = "data/dataset/encoders.pkl"
        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)
        logger.info(f"Encoders saved: {encoders_path}")
    
    logger.info(f"\nPipeline completed!")
    logger.info(f"Ready for model training!")
