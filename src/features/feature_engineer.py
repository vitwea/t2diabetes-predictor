"""
FEATURE ENGINEER - NHANES Diabetes Data

Creates engineered features from raw NHANES data
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NHANESFeatureEngineer:
    """Create features from NHANES diabetes data"""

    COLUMN_MAP = {
        'systolic_bp': 'sbp_1',
        'diastolic_bp': 'dbp_1',
        'glucose': 'glucose_mgdl',
        'hba1c': 'hba1c_percent',
        'insulin': 'insulin_uUml',
        'triglycerides': 'triglycerides_mgdl',
        'total_cholesterol': 'chol_total_mgdl',
        'hdl': 'hdl_mgdl',
        'ldl': 'ldl_mgdl',
        'waist_circumference': 'waist_cm',
        'height': 'height_cm',
        'bmi': 'bmi',
        'carbohydrate': 'carbs_g',
        'fat': 'fat_g',
        'protein': 'protein_g',
    }

    @staticmethod
    def engineer_features(df):
        """Create all features from original dataset"""
        
        logger.info("\n" + "="*70)
        logger.info("FEATURE ENGINEERING: NHANES DIABETES DATA")
        logger.info("="*70)

        df = df.copy()
        cols_before = df.shape[1]
        features_created = 0

        logger.info(f"\nBefore: {cols_before} features")
        logger.info("Creating features...\n")

        def col(key):
            return NHANESFeatureEngineer.COLUMN_MAP.get(key)

        # ===== 1-4: BLOOD PRESSURE INDICES =====
        logger.info("\n[1-4] Blood Pressure Indices:")
        
        if col('systolic_bp') in df.columns:
            df['BP_SYSTOLIC_MEAN'] = df[col('systolic_bp')]
            logger.info(f" BP_SYSTOLIC_MEAN")
            features_created += 1

        if col('diastolic_bp') in df.columns:
            df['BP_DIASTOLIC_MEAN'] = df[col('diastolic_bp')]
            logger.info(f" BP_DIASTOLIC_MEAN")
            features_created += 1

        if col('systolic_bp') in df.columns and col('diastolic_bp') in df.columns:
            df['MEAN_ARTERIAL_PRESSURE'] = (
                (df[col('systolic_bp')] + 2 * df[col('diastolic_bp')]) / 3
            )
            logger.info(f" MEAN_ARTERIAL_PRESSURE")
            features_created += 1

            df['PULSE_PRESSURE'] = df[col('systolic_bp')] - df[col('diastolic_bp')]
            logger.info(f" PULSE_PRESSURE")
            features_created += 1

        # ===== 5-6: INSULIN RESISTANCE =====
        logger.info("\n[5-6] Insulin Resistance Markers:")
        
        if col('insulin') in df.columns and col('glucose') in df.columns:
            df['HOMA_IR'] = (df[col('insulin')] * df[col('glucose')]) / 405
            logger.info(f" HOMA_IR")
            features_created += 1

            insulin_safe = df[col('insulin')].replace(0, np.nan)
            glucose_safe = df[col('glucose')].replace(0, np.nan)
            df['QUICKI'] = 1 / (np.log(insulin_safe) + np.log(glucose_safe))
            logger.info(f" QUICKI")
            features_created += 1

        # ===== 7-8: ANTHROPOMETRIC =====
        logger.info("\n[7-8] Anthropometric Ratios:")
        
        if col('waist_circumference') in df.columns and col('height') in df.columns:
            df['WAIST_HEIGHT_RATIO'] = (
                df[col('waist_circumference')] / df[col('height')]
            )
            logger.info(f" WAIST_HEIGHT_RATIO")
            features_created += 1

        if col('bmi') in df.columns and col('waist_circumference') in df.columns:
            df['BMI_WAIST_RATIO'] = df[col('bmi')] / df[col('waist_circumference')]
            logger.info(f" BMI_WAIST_RATIO")
            features_created += 1

        # ===== 9-12: GLUCOSE & LIPID MARKERS =====
        logger.info("\n[9-12] Glucose & Lipid Markers:")
        
        if col('glucose') in df.columns and col('hba1c') in df.columns:
            df['GLUCOSE_HBAIC_RATIO'] = df[col('glucose')] / df[col('hba1c')]
            logger.info(f" GLUCOSE_HBAIC_RATIO")
            features_created += 1

        if col('triglycerides') in df.columns and col('total_cholesterol') in df.columns:
            df['TG_CHOL_RATIO'] = df[col('triglycerides')] / df[col('total_cholesterol')]
            logger.info(f" TG_CHOL_RATIO")
            features_created += 1

        # ===== HDL-DEPENDENT FEATURES (from original data) =====
        if col('hdl') in df.columns:
            logger.info("\n[10-12] HDL-DEPENDENT Lipid Ratios (from original dataset):")
            
            # 10. TG-HDL Ratio
            if col('triglycerides') in df.columns:
                df['TG_HDL_RATIO'] = df[col('triglycerides')] / df[col('hdl')]
                logger.info(f" TG_HDL_RATIO")
                features_created += 1

            # 11. TC-HDL Ratio
            if col('total_cholesterol') in df.columns:
                df['TC_HDL_RATIO'] = df[col('total_cholesterol')] / df[col('hdl')]
                logger.info(f" TC_HDL_RATIO")
                features_created += 1

            # 12. LDL-HDL Ratio
            if col('ldl') in df.columns:
                df['LDL_HDL_RATIO'] = df[col('ldl')] / df[col('hdl')]
                logger.info(f" LDL_HDL_RATIO")
                features_created += 1

            # Non-HDL = TC - HDL
            if col('total_cholesterol') in df.columns:
                df['NON_HDL_CHOLESTEROL'] = df[col('total_cholesterol')] - df[col('hdl')]
                logger.info(f" NON_HDL_CHOLESTEROL")
                features_created += 1
        else:
            logger.info("\n[10-13] Skip: HDL-dependent ratios (no HDL in dataset)")

        # ===== 13-15: ADVANCED LIPIDS =====
        logger.info("\n[13-15] Advanced Lipid Indices:")
        
        if col('triglycerides') in df.columns and col('glucose') in df.columns:
            df['TRIGLYCERIDE_GLUCOSE'] = np.log(
                (df[col('triglycerides')] * df[col('glucose')]) / 2
            )
            logger.info(f" TRIGLYCERIDE_GLUCOSE (TyG index)")
            features_created += 1

        if all([col(c) in df.columns for c in ['triglycerides', 'glucose', 'waist_circumference', 'height']]):
            tyg = np.log((df[col('triglycerides')] * df[col('glucose')]) / 2)
            whr = df[col('waist_circumference')] / df[col('height')]
            df['TRIGLYCERIDE_GLUCOSE_WAIST'] = tyg * whr
            logger.info(f" TRIGLYCERIDE_GLUCOSE_WAIST (TyG-WHtR)")
            features_created += 1

        # ===== 16-19: DIETARY COMPOSITION =====
        logger.info("\n[16-19] Dietary Composition:")
        
        if all([col(c) in df.columns for c in ['carbohydrate', 'fat', 'protein']]):
            carb_cal = df[col('carbohydrate')] * 4
            fat_cal = df[col('fat')] * 9
            protein_cal = df[col('protein')] * 4
            total_cal = carb_cal + fat_cal + protein_cal

            df['CARB_FAT_RATIO'] = carb_cal / fat_cal
            logger.info(f" CARB_FAT_RATIO")
            features_created += 1

            df['CARB_PERCENTAGE'] = (carb_cal / total_cal) * 100
            logger.info(f" CARB_PERCENTAGE")
            features_created += 1

            df['FAT_PERCENTAGE'] = (fat_cal / total_cal) * 100
            logger.info(f" FAT_PERCENTAGE")
            features_created += 1

            df['PROTEIN_PERCENTAGE'] = (protein_cal / total_cal) * 100
            logger.info(f" PROTEIN_PERCENTAGE")
            features_created += 1

        # ===== 20: CARDIOVASCULAR =====
        logger.info("\n[20] Cardiovascular Stress:")
        
        if col('systolic_bp') in df.columns and col('diastolic_bp') in df.columns:
            df['SYSTOLIC_DIASTOLIC_RATIO'] = (
                df[col('systolic_bp')] / df[col('diastolic_bp')]
            )
            logger.info(f" SYSTOLIC_DIASTOLIC_RATIO")
            features_created += 1

        # ===== 21: METABOLIC SYNDROME =====
        logger.info("\n[21] Metabolic Syndrome Composite:")
        
        mets_components = []
        
        if col('waist_circumference') in df.columns:
            mets_components.append((df[col('waist_circumference')] > 95).astype(int))
            logger.info(f" • Waist circumference > 95 cm")

        if col('triglycerides') in df.columns:
            mets_components.append((df[col('triglycerides')] > 150).astype(int))
            logger.info(f" • Triglycerides > 150 mg/dL")

        if col('hdl') in df.columns:
            mets_components.append((df[col('hdl')] < 45).astype(int))
            logger.info(f" • HDL < 45 mg/dL")

        if col('systolic_bp') in df.columns and col('diastolic_bp') in df.columns:
            mets_components.append(
                ((df[col('systolic_bp')] >= 130) | (df[col('diastolic_bp')] >= 85)).astype(int)
            )
            logger.info(f" • Blood pressure >= 130/85 mmHg")

        if col('glucose') in df.columns:
            mets_components.append((df[col('glucose')] >= 100).astype(int))
            logger.info(f" • Fasting glucose >= 100 mg/dL")

        if mets_components:
            df['METABOLIC_SYNDROME_SCORE'] = pd.concat(mets_components, axis=1).sum(axis=1)
            logger.info(f" METABOLIC_SYNDROME_SCORE")
            features_created += 1

        # Summary
        cols_after = df.shape[1]
        logger.info(f"\n{'='*70}")
        logger.info(f"Features created: {features_created}")
        logger.info(f"Before: {cols_before}, After: {cols_after}")
        logger.info(f"{'='*70}\n")

        return df

    def engineer_and_save(
        self,
        input_path="./data/final/nhanes_diabetes.parquet",
        output_path="./data/final/nhanes_diabetes_engineered.parquet"
    ):
        """Main entry point"""
        
        logger.info("Starting NHANES feature engineering...")
        df = pd.read_parquet(input_path)
        logger.info(f" Loaded: {df.shape}")

        df_engineered = self.engineer_features(df)
        df_engineered.to_parquet(output_path)
        
        logger.info(f" Saved: {output_path}")
        logger.info(f" Shape: {df_engineered.shape}")
        
        return output_path


if __name__ == "__main__":
    engineer = NHANESFeatureEngineer()
    engineer.engineer_and_save()