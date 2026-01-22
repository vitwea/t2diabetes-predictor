"""
SPLIT DATA & CREATE M1, M2, M3 DATASETS
===============================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from src.utils.logger import get_logger
import os

logger = get_logger("split_data")


def analyze_missing_data(df):
    """Analyze distribution of missing values by tiers"""
    missing_pct = (df.isna().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    tier1 = missing_pct[missing_pct <= 30].index.tolist()
    tier2 = missing_pct[(missing_pct > 30) & (missing_pct <= 60)].index.tolist()
    tier3 = missing_pct[missing_pct > 60].index.tolist()
    
    logger.info(f"\nMissing Data Analysis:")
    logger.info(f" TIER 1 (≤30%, impute):  {len(tier1):2d} variables")
    if tier1:
        logger.info(f"  → {tier1}")
    logger.info(f" TIER 2 (30-60%, check): {len(tier2):2d} variables")
    if tier2:
        logger.info(f"  → {tier2}")
    logger.info(f" TIER 3 (>60%, drop):    {len(tier3):2d} variables")
    if tier3:
        logger.info(f"  → {tier3}")
    
    return tier1, tier2, tier3, missing_pct


def create_m1_dataset(X_train, X_test, y_train, y_test, tier1, tier2, tier3):
    """
    M1: TIER 1 only (≤30% missing) imputed with median
    - TIER 2 & 3: dropped
    - Strategy: Conservative (minimal imputation)
    """

    logger.info("\n" + "="*80)
    logger.info("CREATING M1: TIER 1 MEDIAN (Conservative)")
    logger.info("="*80)
    
    X_train_m1 = X_train[[c for c in X_train.columns if c in tier1]].copy()
    X_test_m1 = X_test[[c for c in X_test.columns if c in tier1]].copy()
    
    # Impute TIER 1 con median (fit onñy oin train)
    imputer = SimpleImputer(strategy='median')
    X_train_m1 = pd.DataFrame(
        imputer.fit_transform(X_train_m1),
        columns=X_train_m1.columns,
        index=X_train_m1.index
    )
    X_test_m1 = pd.DataFrame(
        imputer.transform(X_test_m1),
        columns=X_test_m1.columns,
        index=X_test_m1.index
    )
    
    logger.info(f" Features: {len(X_train_m1.columns)}")
    logger.info(f" Train shape: {X_train_m1.shape}")
    logger.info(f" Test shape: {X_test_m1.shape}")
    logger.info(f" Missing values: {X_train_m1.isna().sum().sum()}")
    
    return X_train_m1, X_test_m1, y_train, y_test


def create_m2_dataset(X_train, X_test, y_train, y_test, tier1, tier2, tier3):
    """
    M2: TIER 1 + TIER 2 (≤60% missing) imputed with median
    - TIER 3: dropped
    - Strategy: Balanced (more data but with more missing)
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING M2: TIER 1 + TIER 2 MEDIAN (Balanced)")
    logger.info("="*80)
    
    features_to_keep = tier1 + tier2
    X_train_m2 = X_train[[c for c in X_train.columns if c in features_to_keep]].copy()
    X_test_m2 = X_test[[c for c in X_test.columns if c in features_to_keep]].copy()
    
    # Impute TIER 1 + TIER 2 con median
    imputer = SimpleImputer(strategy='median')
    X_train_m2 = pd.DataFrame(
        imputer.fit_transform(X_train_m2),
        columns=X_train_m2.columns,
        index=X_train_m2.index
    )
    X_test_m2 = pd.DataFrame(
        imputer.transform(X_test_m2),
        columns=X_test_m2.columns,
        index=X_test_m2.index
    )
    
    logger.info(f" Features: {len(X_train_m2.columns)}")
    logger.info(f" Train shape: {X_train_m2.shape}")
    logger.info(f" Test shape: {X_test_m2.shape}")
    logger.info(f" Missing values: {X_train_m2.isna().sum().sum()}")
    
    return X_train_m2, X_test_m2, y_train, y_test


def create_m3_dataset(X_train, X_test, y_train, y_test, tier1, tier2, tier3):
    """
    M3: TIER 1 ONLY (≤30%) imputed with KNN 
    - TIER 2 & 3: dropped
    - Strategy: Aggressive 
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING M3: TIER 1 KNN (Aggressive/Quality)")
    logger.info("="*80)
    
    X_train_m3 = X_train[[c for c in X_train.columns if c in tier1]].copy()
    X_test_m3 = X_test[[c for c in X_test.columns if c in tier1]].copy()
    
    # KNN imputation (n_neighbors=5, usa 5 vecinos más cercanos)
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_train_m3 = pd.DataFrame(
        imputer.fit_transform(X_train_m3),
        columns=X_train_m3.columns,
        index=X_train_m3.index
    )
    X_test_m3 = pd.DataFrame(
        imputer.transform(X_test_m3),
        columns=X_test_m3.columns,
        index=X_test_m3.index
    )
    
    logger.info(f" Features: {len(X_train_m3.columns)}")
    logger.info(f" Train shape: {X_train_m3.shape}")
    logger.info(f" Test shape: {X_test_m3.shape}")
    logger.info(f" Missing values: {X_train_m3.isna().sum().sum()}")
    
    return X_train_m3, X_test_m3, y_train, y_test


def run_pipeline(input_path, output_dir='./data/processed'):
    """
    Main pipeline:
    1. Load engineered dataset (M0)
    2. Train/test split (80/20, stratificado)
    3. Create M1, M2, M3 with different strategies
    4. Save 6 datasets + metadata
    """
    
    logger.info("\n" + "="*80)
    logger.info("DATA SPLITTING & IMPUTATION PIPELINE")
    logger.info("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== STEP 1: Load ====================
    logger.info("\nSTEP 1: LOADING DATA")
    logger.info("="*80)
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Target distribution: {dict(df['diabetes_dx'].value_counts().sort_index())}")
    
    # ==================== STEP 2: Analyze Missing Data ====================
    logger.info("\nSTEP 2: ANALYZE MISSING DATA")
    logger.info("="*80)
    
    tier1, tier2, tier3, missing_pct = analyze_missing_data(df)
    
    # ==================== STEP 3: Train/Test Split ====================
    logger.info("\nSTEP 3: TRAIN/TEST SPLIT")
    logger.info("="*80)
    
    X = df.drop('diabetes_dx', axis=1)
    y = df['diabetes_dx']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f" Train set: {X_train.shape[0]:,} samples ({100*len(X_train)/len(X):.1f}%)")
    logger.info(f"  Diabetic: {(y_train==1).sum():,} ({100*(y_train==1).mean():.1f}%)")
    logger.info(f"  Non-diabetic: {(y_train==0).sum():,} ({100*(y_train==0).mean():.1f}%)")
    logger.info(f" Test set: {X_test.shape[0]:,} samples ({100*len(X_test)/len(X):.1f}%)")
    logger.info(f"  Diabetic: {(y_test==1).sum():,} ({100*(y_test==1).mean():.1f}%)")
    logger.info(f"  Non-diabetic: {(y_test==0).sum():,} ({100*(y_test==0).mean():.1f}%)")
    
    # ==================== STEP 4: Create M1, M2, M3 ====================
    logger.info("\nSTEP 4: CREATE M1, M2, M3 DATASETS")
    logger.info("="*80)
    
    # M1: Conservative (TIER 1 median)
    X_train_m1, X_test_m1, y_train_m1, y_test_m1 = create_m1_dataset(
        X_train, X_test, y_train, y_test, tier1, tier2, tier3
    )
    
    # M2: Balanced (TIER 1 + TIER 2 median)
    X_train_m2, X_test_m2, y_train_m2, y_test_m2 = create_m2_dataset(
        X_train, X_test, y_train, y_test, tier1, tier2, tier3
    )
    
    # M3: Aggressive (TIER 1 KNN)
    X_train_m3, X_test_m3, y_train_m3, y_test_m3 = create_m3_dataset(
        X_train, X_test, y_train, y_test, tier1, tier2, tier3
    )
    
    # ==================== STEP 5: Save ====================
    logger.info("\nSTEP 5: SAVING DATASETS")
    logger.info("="*80)
    
    # Add target back to X for easier loading
    def save_dataset(X, y, name):
        df_combined = pd.concat([X, y.rename('diabetes_dx')], axis=1)
        path = f"{output_dir}/{name}.parquet"
        df_combined.to_parquet(path, index=False)
        logger.info(f" ✓ {path}")
        return path
    
    save_dataset(X_train_m1, y_train_m1, "m1_train")
    save_dataset(X_test_m1, y_test_m1, "m1_test")
    
    save_dataset(X_train_m2, y_train_m2, "m2_train")
    save_dataset(X_test_m2, y_test_m2, "m2_test")
    
    save_dataset(X_train_m3, y_train_m3, "m3_train")
    save_dataset(X_test_m3, y_test_m3, "m3_test")
    
    # ==================== STEP 6: Summary ====================
    logger.info("\nSUMMARY TABLE")
    logger.info("="*80)
    
    summary = pd.DataFrame({
        'Dataset': ['M1', 'M1', 'M2', 'M2', 'M3', 'M3'],
        'Split': ['Train', 'Test', 'Train', 'Test', 'Train', 'Test'],
        'Rows': [X_train_m1.shape[0], X_test_m1.shape[0], 
                 X_train_m2.shape[0], X_test_m2.shape[0],
                 X_train_m3.shape[0], X_test_m3.shape[0]],
        'Features': [X_train_m1.shape[1], X_test_m1.shape[1],
                     X_train_m2.shape[1], X_test_m2.shape[1],
                     X_train_m3.shape[1], X_test_m3.shape[1]],
        'Strategy': ['TIER 1 Median', 'TIER 1 Median',
                     'TIER 1+2 Median', 'TIER 1+2 Median',
                     'TIER 1 KNN', 'TIER 1 KNN']
    })
    logger.info(f"\n{summary.to_string(index=False)}")
    
    logger.info(f"\n Pipeline complete. Ready for modeling.\n")
    
    return {
        'summary': summary,
        'tier1': tier1,
        'tier2': tier2,
        'tier3': tier3
    }


if __name__ == "__main__":
    run_pipeline(
        input_path='./data/final/nhanes_diabetes_engineered.parquet',
        output_dir='./data/processed'
    )
