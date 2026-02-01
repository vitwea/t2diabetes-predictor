"""
Feature Engineering module
Creates clinically meaningful derived variables from existing features.

Feature Engineering Strategy:
1. Ratio features (combining related measurements)
2. Binning/Categorization (creating clinical risk groups)
3. Interaction features (relationships between variables)
4. Domain-specific metrics (medical indices)

Total features: 9 
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio and proportion features from existing variables.
    
    Features created:
    - cholesterol_ratio: HDL / Total Cholesterol (higher is better)
    - triglyceride_ratio: triglycerides / HDL (lower is better)
    
    Args:
        df: Input dataframe
    Returns:
        DataFrame with new ratio features added
    """
    logger.info("\n[Feature Eng 1/4] Creating ratio features")
    logger.info("-" * 80)
    
    # HDL to Total Cholesterol ratio
    if 'hdl_cholesterol' in df.columns and 'total_cholesterol' in df.columns:
        mask = (df['total_cholesterol'] > 0) & (df['total_cholesterol'].notna())
        df.loc[mask, 'cholesterol_ratio'] = (
            df.loc[mask, 'hdl_cholesterol'] / df.loc[mask, 'total_cholesterol']
        )
        
        logger.info(f" Created cholesterol_ratio (HDL/Total)")
        logger.info(f"   Range: {df['cholesterol_ratio'].min():.3f} - {df['cholesterol_ratio'].max():.3f}")
    
    # Triglyceride to HDL ratio
    if 'triglycerides' in df.columns and 'hdl_cholesterol' in df.columns:
        mask = (df['hdl_cholesterol'] > 0) & (df['hdl_cholesterol'].notna())
        df.loc[mask, 'triglyceride_ratio'] = (
            df.loc[mask, 'triglycerides'] / df.loc[mask, 'hdl_cholesterol']
        )
        
        logger.info(f" Created triglyceride_ratio (Triglycerides/HDL)")
        logger.info(f"   Range: {df['triglyceride_ratio'].min():.3f} - {df['triglyceride_ratio'].max():.3f}")
    
    return df


def create_risk_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinical risk categories from continuous variables.
    
    Categories created:
    - bmi_category: Underweight, Normal, Overweight, Obese
    - bp_category: Normal, Elevated, Hypertension Stage 1/2
    - age_group: Young, Middle-aged, Older
    
    Args:
        df: Input dataframe
    Returns:
        DataFrame with new categorical risk features added
    """
    logger.info("\n[Feature Eng 2/4] Creating risk categories")
    logger.info("-" * 80)
    
    # BMI Categories (WHO classification)
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        logger.info(f" Created bmi_category")
        logger.info(f"   Distribution:\n{df['bmi_category'].value_counts().to_string()}")
    
    # Blood Pressure Categories (AHA/ACC 2017 Guidelines)
    if 'systolic_bp' in df.columns:
        df['bp_category'] = pd.cut(
            df['systolic_bp'],
            bins=[0, 120, 130, 140, float('inf')],
            labels=['Normal', 'Elevated', 'Stage1', 'Stage2']
        )
        
        logger.info(f" Created bp_category")
        logger.info(f"   Distribution:\n{df['bp_category'].value_counts().to_string()}")
    
    # Age Groups
    if 'age_years' in df.columns:
        df['age_group'] = pd.cut(
            df['age_years'],
            bins=[0, 40, 60, float('inf')],
            labels=['Young', 'Middle-aged', 'Older']
        )
        
        logger.info(f" Created age_group")
        logger.info(f"   Distribution:\n{df['age_group'].value_counts().to_string()}")
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features combining related medical measurements.
    
    Features created:
    - metabolic_risk_score: Combined metabolic syndrome indicators (CRITICAL for diabetes)
    - age_bmi_interaction: Age × BMI (risk increases with both)
    
    Args:
        df: Input dataframe
    Returns:
        DataFrame with new interaction features added
    """
    logger.info("\n[Feature Eng 3/4] Creating interaction features")
    logger.info("-" * 80)
    
    # METABOLIC RISK SCORE - CRITICAL FOR DIABETES PREDICTION
    # Combines: glucose, triglycerides, HDL, waist circumference, BP
    if all(col in df.columns for col in ['glucose_value', 'triglycerides', 'hdl_cholesterol', 'waist_cm', 'systolic_bp']):
        metabolic_score = pd.DataFrame(index=df.index)
        
        # Normalize each component to 0-1 scale
        metabolic_score['glucose_norm'] = (df['glucose_value'] - df['glucose_value'].min()) / (df['glucose_value'].max() - df['glucose_value'].min())
        metabolic_score['trigly_norm'] = (df['triglycerides'] - df['triglycerides'].min()) / (df['triglycerides'].max() - df['triglycerides'].min())
        metabolic_score['hdl_norm'] = (df['hdl_cholesterol'] - df['hdl_cholesterol'].min()) / (df['hdl_cholesterol'].max() - df['hdl_cholesterol'].min())
        metabolic_score['waist_norm'] = (df['waist_cm'] - df['waist_cm'].min()) / (df['waist_cm'].max() - df['waist_cm'].min())
        metabolic_score['bp_norm'] = (df['systolic_bp'] - df['systolic_bp'].min()) / (df['systolic_bp'].max() - df['systolic_bp'].min())
        
        # Average across components (higher = worse)
        df['metabolic_risk_score'] = metabolic_score.mean(axis=1)
        
        logger.info(f" Created metabolic_risk_score (CRITICAL)")
        logger.info(f"   Components: Glucose, Triglycerides, HDL, Waist, BP")
        logger.info(f"   Range: {df['metabolic_risk_score'].min():.3f} - {df['metabolic_risk_score'].max():.3f}")
    
    # Age × BMI Interaction
    if all(col in df.columns for col in ['age_years', 'bmi']):
        df['age_bmi_interaction'] = df['age_years'] * df['bmi']
        
        logger.info(f" Created age_bmi_interaction")
        logger.info(f"   Range: {df['age_bmi_interaction'].min():.1f} - {df['age_bmi_interaction'].max():.1f}")
    
    return df


def create_medical_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create medical/clinical indices commonly used in health research.
    
    Features created:
    - waist_height_ratio: Central obesity indicator (FINDRISK validated)
    - glucose_bmi_index: Combined metabolic risk indicator
    
    Args:
        df: Input dataframe
    Returns:
        DataFrame with new medical indices added
    """
    logger.info("\n[Feature Eng 4/4] Creating medical indices")
    logger.info("-" * 80)
    
    # Waist-to-Height Ratio (central obesity indicator)
    if all(col in df.columns for col in ['waist_cm', 'height_cm']):
        mask = (df['height_cm'] > 0) & (df['height_cm'].notna())
        df.loc[mask, 'waist_height_ratio'] = (
            df.loc[mask, 'waist_cm'] / df.loc[mask, 'height_cm']
        )
        
        logger.info(f" Created waist_height_ratio")
        logger.info(f"   Range: {df['waist_height_ratio'].min():.3f} - {df['waist_height_ratio'].max():.3f}")
        logger.info(f"   Clinical cutoff for central obesity: 0.5")
    
    # Glucose-BMI Index (metabolic composite)
    if all(col in df.columns for col in ['glucose_value', 'bmi']):
        df['glucose_bmi_index'] = df['glucose_value'] * df['bmi'] / 1000
        
        logger.info(f" Created glucose_bmi_index")
        logger.info(f"   Range: {df['glucose_bmi_index'].min():.3f} - {df['glucose_bmi_index'].max():.3f}")
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps - OPTIMIZED FOR DIABETES TYPE 2 PREDICTION.
    
    Creates:
    1. Ratio features (2 features)
    2. Risk categories (3 categorical features)
    3. Interaction features (2 features)
    4. Medical indices (2 features)
    
    Total: 9 new features (optimized from 12)
    
    REMOVED FEATURES:
    - waist_bmi_ratio (redundant with waist_height_ratio)
    - cardiovascular_risk (less relevant for diabetes prediction)
    - creatinine_age_index (kidney complications, not diabetes onset)
    
    Args:
        df: Input dataframe
    Returns:
        DataFrame with all engineered features added
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FEATURE ENGINEERING (OPTIMIZED FOR DIABETES TYPE 2)")
    logger.info("=" * 80)
    
    initial_cols = len(df.columns)
    
    # Apply all feature engineering steps
    df = create_ratio_features(df)
    df = create_risk_categories(df)
    df = create_interaction_features(df)
    df = create_medical_indices(df)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Initial columns: {initial_cols}")
    logger.info(f"Final columns: {len(df.columns)}")
    logger.info(f"New features created: {len(df.columns) - initial_cols}")
    
    # List all new features
    new_features = [
        'cholesterol_ratio', 'triglyceride_ratio',
        'bmi_category', 'bp_category', 'age_group',
        'metabolic_risk_score', 'age_bmi_interaction',
        'waist_height_ratio', 'glucose_bmi_index'
    ]
    
    actual_new_features = [col for col in new_features if col in df.columns]
    
    logger.info(f"\nNew features added ({len(actual_new_features)}):")
    for i, feature in enumerate(actual_new_features, 1):
        if feature == 'metabolic_risk_score':
            logger.info(f" {i:2d}. {feature} ⭐ CRITICAL FOR DIABETES")
        else:
            logger.info(f" {i:2d}. {feature}")
    
    return df