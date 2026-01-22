"""
FEATURE ENGINEERING
==============================================

Strategy: Create ONLY features that:
1. Have strong clinical evidence
2. Add predictive value (not redundant)
3. Work across all 3 models

Features to CREATE (7 total):
  avg_sbp / avg_dbp - Reduce noise from 2 readings
  pulse_pressure - Cardiovascular health marker
  mean_arterial_pressure - Combined BP metric
  non_hdl - Atherogenic lipids (TC - HDL)
  whr - Waist-to-height ratio (size-adjusted obesity)
  htn_flag - Hypertension diagnosis
  ms_score - Metabolic syndrome components

Features to SKIP (not enough value):
  obesity_class - Trees handle categorization automatically
  homa_ir - 63% missing, use only in Model 3
  tg_hdl_ratio - 71% missing, use only in Model 3
  high_insulin - Redundant (insulin already in data)
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger

logger = get_logger("feature_engineering_minimal")


def fix_target_variable(df):
    """Convert diabetes_dx: 1.0→1 (diabetic), 2.0→0 (non-diabetic)"""
    logger.info("="*80)
    logger.info("STEP 1: FIX TARGET VARIABLE")
    logger.info("="*80)
    
    df = df.copy()
    
    before = df['diabetes_dx'].value_counts().sort_index()
    logger.info(f"\nBefore: {dict(before)}")
    
    # 1.0 = Diabetic → 1, 2.0 = Non-diabetic → 0
    df['diabetes_dx'] = (df['diabetes_dx'] == 1).astype(int)
    
    logger.info(f"\nAfter fix:")
    logger.info(f"  1 (Diabetic):     {(df['diabetes_dx']==1).sum():6,} ({100*(df['diabetes_dx']==1).mean():5.1f}%)")
    logger.info(f"  0 (Non-diabetic): {(df['diabetes_dx']==0).sum():6,} ({100*(df['diabetes_dx']==0).mean():5.1f}%)")
    
    imbalance = (df['diabetes_dx']==0).sum() / (df['diabetes_dx']==1).sum()
    logger.info(f"  Class imbalance: {imbalance:.2f}:1")
    logger.info(f"  Target variable fixed\n")
    
    return df


def create_minimal_features(df):
    """Create 7 essential features with clinical evidence"""
    logger.info("="*80)
    logger.info("STEP 2: CREATE ESSENTIAL FEATURES")
    logger.info("="*80)
    
    df = df.copy()
    
    # 1-2: Average Blood Pressure (reduce measurement noise)
    logger.info("\n[1-2] Blood Pressure Averaging:")
    
    if 'sbp_1' in df.columns and 'sbp_2' in df.columns:
        df['avg_sbp'] = (df['sbp_1'] + df['sbp_2']) / 2
        logger.info("  avg_sbp = (sbp_1 + sbp_2) / 2")
    
    if 'dbp_1' in df.columns and 'dbp_2' in df.columns:
        df['avg_dbp'] = (df['dbp_1'] + df['dbp_2']) / 2
        logger.info("  avg_dbp = (dbp_1 + dbp_2) / 2")
        logger.info("    Reason: Reduces measurement noise (2 readings average better than 1)")
    
    # 3: Pulse Pressure (SBP - DBP)
    logger.info("\n[3] Pulse Pressure:")
    
    if 'avg_sbp' in df.columns and 'avg_dbp' in df.columns:
        df['pulse_pressure'] = df['avg_sbp'] - df['avg_dbp']
        logger.info("  pulse_pressure = avg_sbp - avg_dbp")
        logger.info("    Reason: Cardiovascular health marker (artery stiffness)")
    
    # 4: Mean Arterial Pressure (MAP)
    logger.info("\n[4] Mean Arterial Pressure:")
    
    if 'avg_sbp' in df.columns and 'avg_dbp' in df.columns:
        df['map'] = (df['avg_sbp'] + 2 * df['avg_dbp']) / 3
        logger.info("  map = (avg_sbp + 2*avg_dbp) / 3")
        logger.info("    Reason: Clinical measure of average pressure (used in diagnosis)")
    
    # 5: Non-HDL Cholesterol (TC - HDL)
    logger.info("\n[5] Non-HDL Cholesterol:")
    
    if 'chol_total_mgdl' in df.columns and 'hdl_mgdl' in df.columns:
        df['non_hdl'] = df['chol_total_mgdl'] - df['hdl_mgdl']
        valid = df['non_hdl'].notna().sum()
        logger.info(f"  non_hdl = chol_total - hdl ({valid:,} valid)")
        logger.info("    Reason: Better predictor than TC alone (atherogenic lipids)")
    
    # 6: Waist-to-Height Ratio (WHtR)
    logger.info("\n[6] Waist-to-Height Ratio:")
    
    if 'waist_cm' in df.columns and 'height_cm' in df.columns:
        df['whr'] = df['waist_cm'] / df['height_cm']
        valid = df['whr'].notna().sum()
        logger.info(f"  whr = waist_cm / height_cm ({valid:,} valid)")
        logger.info("    Reason: Size-adjusted obesity measure (better than waist alone)")
    
    # 7: Hypertension Flag (ACC/AHA 2017 guideline)
    logger.info("\n[7] Hypertension Flag:")
    
    if 'avg_sbp' in df.columns and 'avg_dbp' in df.columns:
        df['htn_flag'] = ((df['avg_sbp'] >= 130) | (df['avg_dbp'] >= 85)).astype(int)
        n_htn = df['htn_flag'].sum()
        pct_htn = 100 * df['htn_flag'].mean()
        logger.info(f"  htn_flag = 1 if avg_sbp≥130 or avg_dbp≥85 ({n_htn:,}, {pct_htn:.1f}%)")
        logger.info("    Reason: Binary diagnosis flag (clinical utility)")
    
    logger.info(f"\nFEATURES CREATED\n")
    return df


def strategic_imputation(df):
    """Impute only TIER 1 & TIER 2 variables (keep TIER 3 as-is for Model 2/3)"""
    logger.info("="*80)
    logger.info("STEP 3: STRATEGIC IMPUTATION")
    logger.info("="*80)
    
    df = df.copy()
    
    # Analyze missing
    missing_pct = (df.isna().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    logger.info("\nMissing data analysis:")
    tier1 = missing_pct[missing_pct <= 30].index.tolist()
    tier2 = missing_pct[(missing_pct > 30) & (missing_pct <= 60)].index.tolist()
    tier3 = missing_pct[missing_pct > 60].index.tolist()
    
    logger.info(f"  TIER 1 (≤30%, impute):  {len(tier1):2d} variables")
    logger.info(f"  TIER 2 (30-60%, check): {len(tier2):2d} variables")
    logger.info(f"  TIER 3 (>60%, skip):    {len(tier3):2d} variables (for Model 2/3)")
    
    # Impute TIER 1 with median (robust, simple)
    if tier1:
        logger.info(f"\n→ Median imputation for TIER 1 ({len(tier1)} vars)...")
        imputer = SimpleImputer(strategy='median')
        df[tier1] = imputer.fit_transform(df[tier1])
        logger.info(f"  ✓ Complete")
    
    # TIER 2: Only impute if strategically important
    # For now: leave as-is (will be handled in model-specific datasets)
    logger.info(f"\n→ TIER 2 & TIER 3 left as-is")
    logger.info(f"  (Will be handled separately in Model 1/2/3 subsets)")
    
    final_missing = df.isna().sum().sum()
    logger.info(f"\nImputation complete ({final_missing:,} missing cells remain in TIER 2/3)\n")
    
    return df


def run_pipeline(input_path, output_path):
    """Main pipeline"""
    logger.info("\n" + "="*80)
    logger.info("MINIMAL FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)
    
    # Load
    logger.info(f"\nLoading: {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Execute
    df = fix_target_variable(df)
    df = create_minimal_features(df)
    df = strategic_imputation(df)
    
    # Save
    logger.info("="*80)
    logger.info("SAVING ENGINEERED DATASET")
    logger.info("="*80)
    
    df.to_parquet(output_path, index=False)
    
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Rows: {df.shape[0]:,}")
    logger.info(f"  Columns: {df.shape[1]}")
    logger.info(f"  Features: {sorted([c for c in df.columns if c != 'diabetes_dx'])}")
    logger.info(f"  Missing: {(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
    
    logger.info(f"\nSaved: {output_path}")
    
    return df


if __name__ == "__main__":
    df_engineered = run_pipeline(
        input_path='./data/final/nhanes_diabetes.parquet',
        output_path='./data/final/nhanes_diabetes_engineered.parquet'
    )