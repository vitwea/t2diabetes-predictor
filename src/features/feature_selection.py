"""
FEATURE SELECTION MODULE
Selects TOP 20 features from dataset
Can be imported and used independently
"""

import pandas as pd
import pickle
import os
from src.utils.logger import get_logger

logger = get_logger("feature_selection")

# ════════════════════════════════════════════════════════════════════════════════
# TOP 20 FEATURES - FINAL V2
# ════════════════════════════════════════════════════════════════════════════════

TOP_FEATURES = [
    'age_years',
    'waist_cm',
    'waist_height_ratio',
    'weight_kg',
    'hypertension',
    'bmi',
    'cholesterol_ratio',
    'hdl_cholesterol',
    'systolic_bp',
    'triglyceride_ratio',
    'total_cholesterol',
    'triglycerides',
    'age_bmi_interaction',
    'age_group',
    'ethnicity',
    'creatinine',
    'height_cm',
    'income_poverty_ratio',
    'diastolic_bp',
    'sleep_hours'
]

# ════════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def select_top_features(df: pd.DataFrame, features: list = None) -> tuple:
    """
    Select only TOP 20 features from dataset.
    
    Args:
        df: DataFrame with all features
        features: List of features to select (default: TOP_FEATURES)
    
    Returns:
        Tuple of (df_selected, selected_features)
        - df_selected: DataFrame with only TOP 20 features + target
        - selected_features: List of selected feature names
    
    Example:
        from feature_selection import select_top_features
        
        df_selected, features = select_top_features(df)
        # or
        df_selected, features = select_top_features(df, features=['age_years', 'bmi', ...])
    """
    
    if features is None:
        features = TOP_FEATURES
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE SELECTION: TOP FEATURES")
    logger.info("="*80)
    
    logger.info(f"\n[1/4] Verifying features")
    
    # Check which features exist
    missing = [f for f in features if f not in df.columns]
    
    if missing:
        logger.warning(f"Missing features: {missing}")
        logger.info(f"Removing missing features from selection...")
        features = [f for f in features if f in df.columns]
        logger.info(f"Using {len(features)} features instead of {len(TOP_FEATURES)}")
    else:
        logger.info(f"All {len(features)} features present")
    
    # Verify target exists
    if 'diabetes_risk' not in df.columns:
        logger.error("Target 'diabetes_risk' not found!")
        raise ValueError("Target column 'diabetes_risk' not found")
    
    logger.info(f"\n[2/4] Selecting features")
    
    # Select TOP + target
    df_selected = df[features + ['diabetes_risk']].copy()
    
    logger.info(f"Selected {len(features)} features + target")
    logger.info(f"Total columns: {len(df_selected.columns)}")
    
    logger.info(f"\n[3/4] Data quality check")
    
    # Missing values
    missing_vals = df_selected.isnull().sum().sum()
    if missing_vals > 0:
        logger.warning(f"Missing values: {missing_vals}")
    else:
        logger.info(f"No missing values")
    
    # Target distribution
    target_dist = df_selected['diabetes_risk'].value_counts()
    logger.info(f"Target distribution: {target_dist.to_dict()}")
    
    logger.info(f"\n[4/4] Reduction statistics")
    
    original_cols = len(df.columns)
    selected_cols = len(df_selected.columns)
    removed_cols = original_cols - selected_cols
    reduction_pct = (removed_cols / original_cols) * 100
    
    logger.info(f"Original columns:  {original_cols}")
    logger.info(f"Selected columns:  {selected_cols}")
    logger.info(f"Removed columns:   {removed_cols} ({reduction_pct:.1f}%)")
    
    logger.info("\n" + "="*80)
    logger.info(f"Feature selection completed!")
    logger.info("="*80)
    
    return df_selected, features

# ════════════════════════════════════════════════════════════════════════════════
# SAVE FEATURES FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def save_features(features: list, output_path: str = 'data/models/top_features_final.pkl') -> None:
    """
    Save selected features list to pickle file.
    
    Args:
        features: List of feature names
        output_path: Path where to save the pickle file
    
    Example:
        from feature_selection import save_features
        save_features(selected_features)
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    logger.info(f"Features saved: {output_path}")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION 
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    logger.info("Loading dataset from Parquet")
    df = pd.read_parquet("data/nhanes_data/dataset_engineered_encoded.parquet")
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    logger.info("Running feature selection")
    
    # Select TOP features
    df_selected, selected_features = select_top_features(df)
    
    logger.info("Saving results")
    
    # Save features list
    save_features(selected_features)
    
    logger.info(f"\nFeature selection completed!")

