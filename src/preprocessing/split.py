import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("preprocessing.split")


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
):
    """
    Split dataset into train and test sets.
    """

    logger.info("=" * 80)
    logger.info("TRAIN / TEST SPLIT")
    logger.info("=" * 80)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

    return X_train, X_test, y_train, y_test
