"""
FEATURE ENGINEERING - MINIMAL & CLINICAL
Features to CREATE (7 total):
- avg_sbp/avg_dbp: Reduce noise
- pulse_pressure: Cardiovascular marker
- map: Mean arterial pressure
- non_hdl: Atherogenic lipids
- whr: Waist-to-height ratio
- htn_flag: Hypertension diagnosis
"""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")

def fix_target_variable(df):
    """Convert diabetes_dx: 1.0→1, 2.0→0"""
    logger.info("="*80)
    logger.info("STEP 1: FIX TARGET VARIABLE")
    logger.info("="*80)
    
    df = df.copy()
    df['diabetes_dx'] = (df['diabetes_dx'] == 1).astype(int)
    
    logger.info(f" 1 (Diabetic): {(df['diabetes_dx']==1).sum():6,} ({100*(df['diabetes_dx']==1).mean():5.1f}%)")
    logger.info(f" 0 (Non-diabetic): {(df['diabetes_dx']==0).sum():6,} ({100*(df['diabetes_dx']==0).mean():5.1f}%)")
    imbalance = (df['diabetes_dx']==0).sum() / (df['diabetes_dx']==1).sum()
    logger.info(f" Class imbalance: {imbalance:.2f}:1\n")
    
    return df

def create_minimal_features(df):
    """Create 7 clinical features"""
    logger.info("="*80)
    logger.info("STEP 2: CREATE ESSENTIAL FEATURES")
    logger.info("="*80)
    
    df = df.copy()
    
    # Blood Pressure Averaging
    if 'sbp_1' in df.columns and 'sbp_2' in df.columns:
        df['avg_sbp'] = (df['sbp_1'] + df['sbp_2']) / 2
    if 'dbp_1' in df.columns and 'dbp_2' in df.columns:
        df['avg_dbp'] = (df['dbp_1'] + df['dbp_2']) / 2
    
    # Pulse Pressure
    if 'avg_sbp' in df.columns and 'avg_dbp' in df.columns:
        df['pulse_pressure'] = df['avg_sbp'] - df['avg_dbp']
    
    # MAP
    if 'avg_sbp' in df.columns and 'avg_dbp' in df.columns:
        df['map'] = (df['avg_sbp'] + 2 * df['avg_dbp']) / 3
    
    # Non-HDL
    if 'chol_total_mgdl' in df.columns and 'hdl_mgdl' in df.columns:
        df['non_hdl'] = df['chol_total_mgdl'] - df['hdl_mgdl']
    
    # WHtR
    if 'waist_cm' in df.columns and 'height_cm' in df.columns:
        df['whr'] = df['waist_cm'] / df['height_cm']
    
    # HTN Flag
    if 'avg_sbp' in df.columns and 'avg_dbp' in df.columns:
        df['htn_flag'] = ((df['avg_sbp'] >= 130) | (df['avg_dbp'] >= 85)).astype(int)
    
    logger.info(f"FEATURES CREATED\n")
    return df

def run_pipeline(input_path, output_path):
    """Main pipeline: Load → Fix → Create → Save"""
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*80 + "\n")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    df = fix_target_variable(df)
    df = create_minimal_features(df)
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    logger.info(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    return df

if __name__ == "__main__":
    run_pipeline(
        './data/final/nhanes_diabetes.parquet',
        './data/final/nhanes_diabetes_engineered.parquet'
    )
