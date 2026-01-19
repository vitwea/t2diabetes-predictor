import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# ===========================
# Logging
# ===========================

logger = get_logger(__name__)

logger.info("Starting data modeling step...")


# ===========================
# LOAD DATA
# ===========================

def load_data(filepath):
    """Load NHANES data from parquet file"""
    logger.info("="*70)
    logger.info("LOADING DATA")
    logger.info("="*70)
    
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Columns: {df.shape[1]}")
        logger.info(f"  - Rows: {df.shape[0]:,}")
        return df
    except FileNotFoundError:
        logger.error(f"✗ File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading file: {e}")
        raise

# ===========================
# ANALYZE CLASS IMBALANCE
# ===========================

def analyze_class_imbalance(df, target_col='diabetes_dx'):
    """Analyze class imbalance ratio"""
    logger.info("\n" + "="*70)
    logger.info("CLASS IMBALANCE ANALYSIS")
    logger.info("="*70)
    
    value_counts = df[target_col].value_counts().sort_index()
    
    logger.info(f"\nClass distribution:")
    for label, count in value_counts.items():
        pct = count / len(df) * 100
        label_name = "Positive (Diabetes)" if label == 1 else "Negative (No Diabetes)"
        logger.info(f"  {label}: {count:,} ({pct:.2f}%) - {label_name}")
    
    count_positive = value_counts.get(1, 0)
    count_negative = value_counts.get(0, 0)
    
    if count_positive == 0 or count_negative == 0:
        logger.warning("✗ One class has no samples!")
        return None
    
    imbalance_ratio = count_negative / count_positive
    
    logger.info(f"\nImbalance ratio: {imbalance_ratio:.2f}:1 (Negative:Positive)")
    logger.info(f"  - Severity: ", end="")
    if imbalance_ratio < 1.5:
        logger.info("Low (well-balanced)")
    elif imbalance_ratio < 3:
        logger.info("Moderate")
    elif imbalance_ratio < 10:
        logger.info("High")
    else:
        logger.info("Severe")
    
    return imbalance_ratio, count_positive, count_negative

# ===========================
# TRAIN/TEST SPLIT (STRATIFIED)
# ===========================

def stratified_train_test_split(df, target_col='diabetes_dx', test_size=0.2, random_state=42):
    """
    Split data with stratification to maintain class distribution
    
    Args:
        df: DataFrame
        target_col: Target column
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("\n" + "="*70)
    logger.info("STRATIFIED TRAIN/TEST SPLIT")
    logger.info("="*70)
    
    logger.info(f"\nSplitting with stratification:")
    logger.info(f"  - Test size: {test_size*100:.0f}%")
    logger.info(f"  - Random state: {random_state}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    logger.info(f"\nTraining set: {len(X_train):,} samples")
    logger.info(f"  - Positive: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
    logger.info(f"  - Negative: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
    
    logger.info(f"\nTest set: {len(X_test):,} samples")
    logger.info(f"  - Positive: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")
    logger.info(f"  - Negative: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test

# ===========================
# HANDLE CLASS IMBALANCE: SMOTE
# ===========================

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique)
    
    Creates synthetic samples of minority class to balance dataset
    Only applied to training set to avoid data leakage
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
    
    Returns:
        X_train_smote, y_train_smote
    """
    logger.info("\n" + "="*70)
    logger.info("APPLY SMOTE (Synthetic Minority Oversampling)")
    logger.info("="*70)
    
    logger.info(f"\nBefore SMOTE:")
    logger.info(f"  - Total samples: {len(X_train):,}")
    logger.info(f"  - Positive: {(y_train == 1).sum():,}")
    logger.info(f"  - Negative: {(y_train == 0).sum():,}")
    logger.info(f"  - Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")
    
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    logger.info(f"\nAfter SMOTE:")
    logger.info(f"  - Total samples: {len(X_train_smote):,}")
    logger.info(f"  - Positive: {(y_train_smote == 1).sum():,}")
    logger.info(f"  - Negative: {(y_train_smote == 0).sum():,}")
    logger.info(f"  - Imbalance ratio: {(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.2f}:1")
    logger.info(f"  - Synthetic samples created: {len(X_train_smote) - len(X_train):,}")
    
    return X_train_smote, y_train_smote

# ===========================
# HANDLE CLASS IMBALANCE: COMBINED RESAMPLING
# ===========================

def apply_combined_resampling(X_train, y_train, random_state=42):
    """
    Apply combined resampling: Undersampling + Oversampling
    
    More conservative approach than pure SMOTE
    - Undersamples majority class slightly
    - Oversamples minority class moderately
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
    
    Returns:
        X_train_resampled, y_train_resampled
    """
    logger.info("\n" + "="*70)
    logger.info("APPLY COMBINED RESAMPLING (Under + Over)")
    logger.info("="*70)
    
    logger.info(f"\nBefore resampling:")
    logger.info(f"  - Total samples: {len(X_train):,}")
    logger.info(f"  - Positive: {(y_train == 1).sum():,}")
    logger.info(f"  - Negative: {(y_train == 0).sum():,}")
    logger.info(f"  - Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")
    
    # Undersampling: reduce majority class to 60% of original
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=random_state)
    
    # Oversampling: increase minority class to 80% of majority
    over = SMOTE(sampling_strategy=0.8, random_state=random_state, k_neighbors=5)
    
    pipeline = ImbPipeline([
        ('under', under),
        ('over', over)
    ])
    
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
    
    logger.info(f"\nAfter combined resampling:")
    logger.info(f"  - Total samples: {len(X_train_resampled):,}")
    logger.info(f"  - Positive: {(y_train_resampled == 1).sum():,}")
    logger.info(f"  - Negative: {(y_train_resampled == 0).sum():,}")
    logger.info(f"  - Imbalance ratio: {(y_train_resampled == 0).sum() / (y_train_resampled == 1).sum():.2f}:1")
    logger.info(f"  - Total change: {len(X_train_resampled) - len(X_train):,} samples")
    
    return X_train_resampled, y_train_resampled

# ===========================
# FEATURE SCALING
# ===========================

def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler
    
    Fit on training set only, transform both train and test
    """
    logger.info("\n" + "="*70)
    logger.info("FEATURE SCALING (StandardScaler)")
    logger.info("="*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"\nScaling completed:")
    logger.info(f"  - Scaler fitted on training set")
    logger.info(f"  - Training features scaled: {X_train_scaled.shape}")
    logger.info(f"  - Test features scaled: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

# ===========================
# SAVE PREPARED DATA
# ===========================

def save_prepared_data(X_train, X_test, y_train, y_test, output_dir="data/prepared"):
    """Save train/test splits to parquet files"""
    logger.info("\n" + "="*70)
    logger.info("SAVING PREPARED DATA")
    logger.info("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target for easier handling
    train_data = X_train.copy()
    train_data['diabetes_dx'] = y_train.values
    
    test_data = X_test.copy()
    test_data['diabetes_dx'] = y_test.values
    
    # Save to parquet
    train_path = output_path / "X_train.parquet"
    test_path = output_path / "X_test.parquet"
    y_train_path = output_path / "y_train.parquet"
    y_test_path = output_path / "y_test.parquet"
    
    train_data.to_parquet(train_path)
    test_data.to_parquet(test_path)
    y_train.to_frame().to_parquet(y_train_path)
    y_test.to_frame().to_parquet(y_test_path)
    
    logger.info(f"\n✓ Files saved to {output_path}:")
    logger.info(f"  - X_train.parquet: {train_path}")
    logger.info(f"  - X_test.parquet: {test_path}")
    logger.info(f"  - y_train.parquet: {y_train_path}")
    logger.info(f"  - y_test.parquet: {y_test_path}")

# ===========================
# MAIN PIPELINE
# ===========================

def prepare_data_for_modeling(
    data_path="C:/Users/pablo/Desktop/t2diabetes-predictor/data/final/nhanes_diabetes.parquet",
    target_col='diabetes_dx',
    test_size=0.2,
    resampling_strategy='smote',  # 'smote' or 'combined'
    random_state=42
):
    """
    Complete data preparation pipeline:
    1. Load data
    2. Analyze class imbalance
    3. Stratified train/test split
    4. Apply resampling (SMOTE or combined)
    5. Scale features
    6. Save prepared data
    
    Args:
        data_path: Path to parquet file
        target_col: Target column name
        test_size: Proportion for test set
        resampling_strategy: 'smote' or 'combined'
        random_state: Random seed
    
    Returns:
        Dictionary with train/test data and scaler
    """
    logger.info("\n" + "="*70)
    logger.info("DATA PREPARATION PIPELINE FOR MODELING")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    df = load_data(data_path)
    
    # Step 2: Analyze class imbalance
    imbalance_info = analyze_class_imbalance(df, target_col)
    
    # Step 3: Stratified train/test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df, target_col, test_size, random_state
    )
    
    # Step 4: Apply resampling
    logger.info("\n" + "="*70)
    logger.info("RESAMPLING STRATEGY")
    logger.info("="*70)
    logger.info(f"\nSelected strategy: {resampling_strategy.upper()}")
    
    if resampling_strategy.lower() == 'smote':
        X_train, y_train = apply_smote(X_train, y_train, random_state)
    elif resampling_strategy.lower() == 'combined':
        X_train, y_train = apply_combined_resampling(X_train, y_train, random_state)
    else:
        logger.warning(f"Unknown resampling strategy: {resampling_strategy}")
    
    # Step 5: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 6: Save prepared data
    save_prepared_data(X_train, X_test, y_train, y_test)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PREPARATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"\nFinal shapes:")
    logger.info(f"  - X_train: {X_train_scaled.shape}")
    logger.info(f"  - X_test: {X_test_scaled.shape}")
    logger.info(f"  - y_train: {y_train.shape}")
    logger.info(f"  - y_test: {y_test.shape}")
    logger.info(f"\nLog file: {log_filepath}")
    logger.info("="*70 + "\n")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }

# ===========================
# USAGE
# ===========================

if __name__ == "__main__":
    # Prepare data with SMOTE
    data = prepare_data_for_modeling(
        data_path="C:/Users/pablo/Desktop/t2diabetes-predictor/data/final/nhanes_diabetes.parquet",
        resampling_strategy='smote',  # or 'combined'
        random_state=42
    )
    
    print("\nData preparation successful!")
    print(f"Training set: {data['X_train'].shape}")
    print(f"Test set: {data['X_test'].shape}")