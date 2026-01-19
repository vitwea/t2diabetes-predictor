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
    """Unified data preparation pipeline with Parquet output"""
    
    def __init__(self, data_path, target_col='diabetes_dx', random_state=42):
        self.data_path = data_path
        self.target_col = target_col
        self.random_state = random_state
    
    def run(self, test_size=0.2, strategy='smote'):
        """Complete pipeline: load → clean → split → impute → resample → scale → save"""
        
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
            
            logger.info(f"Cleaned: {df.shape}")
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
            
            logger.info(f"Type conversion: {df.shape}")
            
            # 3. STRATIFIED SPLIT
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state
            )
            
            logger.info(f"✓ Split: train {X_train.shape}, test {X_test.shape}")
            
            # 4. IMPUTE (with inf/nan handling)
            # Replace inf and -inf with NaN before imputing
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
            
            logger.info(f"Imputed NaN values")
            
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
            
            logger.info(f"Resampled ({strategy}): {X_train.shape}")
            logger.info(f"  - Positive: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
            
            # 6. SCALE
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            logger.info(f"Scaled features")
            
            # 7. SAVE TO PARQUET
            logger.info(f"\n{'='*70}")
            logger.info("SAVING PREPARED DATA TO PARQUET")
            logger.info(f"{'='*70}")
            
            # Create output directory
            output_dir = Path(self.data_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create DataFrames with column names
            train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            train_df[self.target_col] = y_train.values if hasattr(y_train, 'values') else y_train
            
            test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            test_df[self.target_col] = y_test.values if hasattr(y_test, 'values') else y_test
            
            # Define paths
            train_path = output_dir / "train_prepared.parquet"
            test_path = output_dir / "test_prepared.parquet"
            
            # Save to parquet
            train_df.to_parquet(train_path)
            test_df.to_parquet(test_path)
            
            logger.info(f"Saved: {train_path}")
            logger.info(f"  Shape: {train_df.shape}")
            logger.info(f"Saved: {test_path}")
            logger.info(f"  Shape: {test_df.shape}")
            
            # Save metadata
            metadata = {
                'feature_names': X_train.columns.tolist(),
                'target_col': self.target_col,
                'n_features': len(X_train.columns),
                'n_samples_train': len(train_df),
                'n_samples_test': len(test_df),
                'class_distribution_train': {
                    'negative': int((y_train==0).sum()),
                    'positive': int((y_train==1).sum())
                },
                'class_distribution_test': {
                    'negative': int((y_test==0).sum()),
                    'positive': int((y_test==1).sum())
                }
            }
            
            metadata_path = output_dir / "prep_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved: {metadata_path}")
            
            logger.info(f"\n{'='*70}")
            logger.info("FINAL SHAPES")
            logger.info(f"{'='*70}")
            logger.info(f"X_train: {X_train_scaled.shape}")
            logger.info(f"X_test: {X_test_scaled.shape}")
            logger.info(f"y_train: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}")
            logger.info(f"y_test: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
            logger.info("="*70 + "\n")
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
                'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
                'scaler': scaler,
                'imputer': imputer,
                'feature_names': X_train.columns.tolist(),
                'train_path': str(train_path),
                'test_path': str(test_path),
                'metadata_path': str(metadata_path)
            }
        
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {str(e)}")
            raise


def prepare_data_for_modeling(
    data_path="./data/final/nhanes_diabetes_engineered.parquet",
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
    
    logger.info("Data preparation successful!")
    logger.info(f"Files saved to: ./data/final/")
    logger.info(f"  - train_prepared.parquet")
    logger.info(f"  - test_prepared.parquet")
    logger.info(f"  - prep_metadata.json")