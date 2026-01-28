"""
Categorical encoding strategies for preprocessing categorical variables.
Implements One-Hot, Ordinal, and Target Encoding approaches.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from src.utils.logger import get_logger

logger = get_logger("encoding")


def encode_onehot(df: pd.DataFrame, cat_cols: list, drop_first: bool = True) -> tuple:
    """
    Apply One-Hot Encoding to categorical variables.
    """

    logger.info("Applying One-Hot Encoding")
    logger.info(f"  Columns: {cat_cols}")
    logger.info(f"  drop_first: {drop_first}")

    # Filter valid columns
    cat_cols = [col for col in cat_cols if col in df.columns]

    if not cat_cols:
        logger.warning(" No valid columns found for One-Hot Encoding")
        return df, None

    # Log unique categories
    logger.info("  Unique values per column:")
    for col in cat_cols:
        logger.info(f"    {col}: {df[col].nunique()} categories")

    # Encoder
    encoder = OneHotEncoder(
        sparse_output=False,
        drop="first" if drop_first else None
    )

    encoded_array = encoder.fit_transform(df[cat_cols])
    feature_names = encoder.get_feature_names_out(cat_cols)

    # Create encoded DF (keep original index)
    df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)

    # Drop original columns and concat WITHOUT resetting index
    df_result = df.drop(columns=cat_cols)
    df_result = pd.concat([df_result, df_encoded], axis=1)

    logger.info(" One-Hot Encoding completed")
    logger.info(f"  New features created: {len(feature_names)}")

    return df_result, encoder


def encode_ordinal(df: pd.DataFrame, cat_cols: list, categories_order: dict = None) -> tuple:
    """
    Apply Ordinal Encoding to categorical variables.
    """

    logger.info("Applying Ordinal Encoding")
    logger.info(f"  Columns: {cat_cols}")

    if categories_order is None:
        categories_order = {}

    # Filter valid columns
    cat_cols = [col for col in cat_cols if col in df.columns]

    if not cat_cols:
        logger.warning(" No valid columns found for Ordinal Encoding")
        return df, None

    categories = []

    for col in cat_cols:
        if col in categories_order:
            # Custom order
            ordered_list = categories_order[col]
            logger.info(f"  {col}: custom order → {ordered_list}")
            categories.append(ordered_list)
        else:
            # Alphabetical order (with warning)
            ordered_list = sorted(df[col].dropna().unique())
            logger.warning(
                f" No explicit order provided for '{col}'. "
                f"Using alphabetical order: {ordered_list}"
            )
            categories.append(ordered_list)

    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    encoded_array = encoder.fit_transform(df[cat_cols])

    df_result = df.copy()
    df_result[cat_cols] = encoded_array

    logger.info(" Ordinal Encoding completed")
    return df_result, encoder


def encode_target(df: pd.DataFrame, cat_cols: list, target_col: str, smoothing: float = 10.0) -> pd.DataFrame:
    """
    Apply Target Encoding (Mean Encoding) to categorical variables.
    """

    logger.info("Applying Target Encoding")
    logger.info(f"  Columns: {cat_cols}")
    logger.info(f"  Target: {target_col}")
    logger.info(f"  Smoothing: {smoothing}")

    logger.warning(" Target Encoding can cause leakage. Use carefully.")

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found.")
        return df

    # Filter valid columns
    cat_cols = [col for col in cat_cols if col in df.columns]

    if not cat_cols:
        logger.warning(" No valid columns found for Target Encoding")
        return df

    df_result = df.copy()
    global_mean = df[target_col].mean()

    logger.info(f"  Global target mean: {global_mean:.4f}")

    for col in cat_cols:
        stats = df.groupby(col)[target_col].agg(["mean", "count"])

        # Smoothed mean
        stats["smoothed_mean"] = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )

        encoding_map = stats["smoothed_mean"].to_dict()

        df_result[col] = df_result[col].map(encoding_map)

        # Handle unseen categories
        missing = df_result[col].isna().sum()
        if missing > 0:
            logger.warning(f"  {missing} unseen categories in '{col}'. Filling with global mean.")
            df_result[col] = df_result[col].fillna(global_mean)

    logger.info(" Target Encoding completed")
    return df_result


def encode_categorical(df: pd.DataFrame,
                       onehot_cols: list = None,
                       ordinal_cols: list = None,
                       target_cols: dict = None) -> tuple:
    """
    Orchestrates all categorical encoding strategies.
    """

    logger.info("=" * 80)
    logger.info("CATEGORICAL ENCODING ORCHESTRATION")
    logger.info("=" * 80)

    if onehot_cols is None:
        onehot_cols = []
    if ordinal_cols is None:
        ordinal_cols = []
    if target_cols is None:
        target_cols = {}

    encoders = {}

    # One-Hot
    if onehot_cols:
        df, enc = encode_onehot(df, onehot_cols)
        encoders["onehot"] = enc

    # Ordinal
    if ordinal_cols:
        df, enc = encode_ordinal(df, ordinal_cols)
        encoders["ordinal"] = enc

    # Target
    if target_cols and "columns" in target_cols and "target" in target_cols:
        df = encode_target(
            df,
            target_cols["columns"],
            target_cols["target"],
            smoothing=target_cols.get("smoothing", 10.0)
        )
        encoders["target"] = "Applied"

    logger.info("=" * 80)
    logger.info("CATEGORICAL ENCODING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df, encoders