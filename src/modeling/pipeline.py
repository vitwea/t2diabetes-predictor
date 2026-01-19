"""
Data Preparation Pipeline
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """Unified data preparation pipeline"""
    
    def __init__(self, data_path, target_col='diabetes_dx', random_state=42):
        self.data_path = data_path
        self.target_col = target_col
        self.random_state = random_state
    
    def run(self, test_size=0.2, strategy='smote'):
        """Complete pipeline: load → clean → split → impute → resample → scale"""
        logger.info("\n" + "="*70)
        logger.info("DATA PREPARATION PIPELINE")
        logger.info("="*70)
        
        try:
            # 1. LOAD & CLEAN
            df = pd.read_parquet(self.data_path)
            logger.info(f"✓ Loaded: {df.shape}")
            
            # Remove NaN target and filter classes
            df = df.dropna(subset=[self.target_col])
            df = df[df[self.target_col].isin([1.0, 2.0])].copy()
            df[self.target_col] = (df[self.target_col] == 1.0).astype(int)
            
            logger.info(f"✓ Cleaned: {df.shape}")
            logger.info(f"  - Positive: {(df[self.target_col]==1).sum():,} ({(df[self.target_col]==1).sum()/len(df)*100:.1f}%)")
            
            # 2. CONVERT TYPES & DROP BAD COLUMNS
            non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
            non_numeric = [c for c in non_numeric if c != self.target_col]
            
            for col in non_numeric:
                converted = pd.to_numeric(df[col], errors='coerce')
                nan_ratio = converted.isna().sum() / len(converted)
                if nan_ratio > 0.5:
                    df = df.drop(columns=[col])
                    logger.info(f"  - Dropped '{col}' ({nan_ratio*100:.0f}% NaN)")
                else:
                    df[col] = converted
            
            logger.info(f"✓ Type conversion: {df.shape}")
            
            # 3. STRATIFIED SPLIT
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state
            )
            logger.info(f"✓ Split: train {X_train.shape}, test {X_test.shape}")
            
            # 4. IMPUTE
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
            logger.info(f"✓ Imputed NaN values")
            
            # 5. RESAMPLE
            if strategy == 'smote':
                resampler = SMOTE(random_state=self.random_state, k_neighbors=5)
                X_train, y_train = resampler.fit_resample(X_train, y_train)
            elif strategy == 'combined':
                pipeline = ImbPipeline([
                    ('under', RandomUnderSampler(sampling_strategy=0.6, random_state=self.random_state)),
                    ('over', SMOTE(sampling_strategy=0.8, random_state=self.random_state, k_neighbors=5))
                ])
                X_train, y_train = pipeline.fit_resample(X_train, y_train)
            
            logger.info(f"✓ Resampled ({strategy}): {X_train.shape}")
            logger.info(f"  - Positive: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
            
            # 6. SCALE
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            logger.info(f"✓ Scaled features")
            logger.info(f"\nFinal shapes:")
            logger.info(f"  - X_train: {X_train_scaled.shape}")
            logger.info(f"  - X_test: {X_test_scaled.shape}")
            logger.info("="*70 + "\n")
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train.values,
                'y_test': y_test.values,
                'scaler': scaler,
                'imputer': imputer,
                'feature_names': X_train.columns.tolist()
            }
        
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {str(e)}")
            raise


def prepare_data_for_modeling(
    data_path="C:/Users/pablo/Desktop/t2diabetes-predictor/data/final/nhanes_diabetes.parquet",
    target_col='diabetes_dx',
    test_size=0.2,
    resampling_strategy='smote',
    random_state=42
):
    """Main entry point"""
    pipeline = DataPipeline(data_path, target_col, random_state)
    return pipeline.run(test_size, resampling_strategy)


if __name__ == "__main__":
    logger.info("Starting data preparation...")
    data = prepare_data_for_modeling(
        resampling_strategy='smote',
        random_state=42
    )
    logger.info("✓ Data preparation successful!")