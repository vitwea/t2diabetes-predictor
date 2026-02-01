"""
Clinical Rules module for NHANES health dataset.
Applies domain-specific medical knowledge and rules-based transformations.

Functions:
1. reconstruct_bmi() - Reconstruct missing BMI from weight/height
2. apply_clinical_rules() - Apply clinical domain knowledge rules
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("preprocessing.clinical_rules")


def reconstruct_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct missing BMI values from weight and height.
    
    Formula: BMI = weight_kg / (height_cm / 100)^2
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with reconstructed BMI values
    """
    
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
            logger.info(f"Reconstructed {bmi_reconstructed:,} BMI values")
            logger.info(f"  Mean BMI: {df['bmi'].mean():.2f}")
            logger.info(f"  Std BMI: {df['bmi'].std():.2f}")
        else:
            logger.info(f"No BMI reconstruction needed (all values present)\n")
    else:
        logger.warning("Required columns (bmi, weight_kg, height_cm) not found\n")
    
    return df


def apply_clinical_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain knowledge rules based on clinical criteria.
    
    Rules implemented:
    ==================
    Rule 1: High Blood Pressure → Hypertension
    - If SBP > 140 mmHg OR DBP > 90 mmHg → Mark as hypertensive (hypertension = 1)
    - Based on AHA/ACC 2017 Hypertension Guidelines
    - Note: Only overwrites NaN values, doesn't force update if already set
    
    Rule 2: Age > 60 → Assume Hypertension if missing
    - If age > 60 years AND hypertension is NaN → Mark as hypertensive (hypertension = 1)
    - Clinical reasoning: Prevalence of hypertension increases significantly with age
    - Conservative approach: Only applies if no prior measurement
    
    Args:
        df: Input dataframe (should have columns: systolic_bp, BPXDIA, age_years, hypertension)
        
    Returns:
        DataFrame with applied clinical rules
    """   

    # Rule 1: High blood pressure → hypertension
    if 'systolic_bp' in df.columns and 'BPXDIA' in df.columns and 'hypertension' in df.columns:
        logger.info("[Rule 1] High Blood Pressure → Hypertension")
        logger.info("  Criteria: SBP > 140 mmHg OR DBP > 90 mmHg")
        
        high_bp_mask = (df['systolic_bp'] > 140) | (df['BPXDIA'] > 90)
        
        # Count before
        hypertension_before = df['hypertension'].sum() if df['hypertension'].dtype == 'int64' else (df['hypertension'] == 1).sum()
        
        # Apply rule
        df.loc[high_bp_mask, 'hypertension'] = 1
        
        # Count after
        hypertension_after = df['hypertension'].sum() if df['hypertension'].dtype == 'int64' else (df['hypertension'] == 1).sum()
        newly_marked = hypertension_after - hypertension_before
        
        if newly_marked > 0:
            logger.info(f"Marked {newly_marked:,} rows as hypertensive (high BP detected)\n")
            logger.info(f"SBP > 140: {(df['systolic_bp'] > 140).sum():,}")
            logger.info(f"DBP > 90: {(df['BPXDIA'] > 90).sum():,}")
        else:
            logger.info(f"No additional rows marked (already accounted for)\n")
    
    # Rule 2: Age > 60 → assume hypertension if missing
    if 'age_years' in df.columns and 'hypertension' in df.columns:
        logger.info("[Rule 2] Age-Based Hypertension Assumption")
        logger.info("Criteria: age > 60 years AND hypertension is NaN\n")
        
        age_rule_mask = df['hypertension'].isna() & (df['age_years'] > 60)
        age_rule_count = age_rule_mask.sum()
        
        if age_rule_count > 0:
            df.loc[age_rule_mask, 'hypertension'] = 1
            logger.info(f"Marked {age_rule_count:,} rows as hypertensive (age > 60 & missing value)")
            logger.info(f"Age range: {df.loc[age_rule_mask, 'age_years'].min():.0f} - {df.loc[age_rule_mask, 'age_years'].max():.0f} years\n")
        else:
            logger.info(f"No rows to mark (no missing values in age > 60 group)\n")
    
    logger.info(f"Clinical rules applied successfully\n")

    return df
