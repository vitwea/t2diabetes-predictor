"""
Categorical encoding strategies for preprocessing categorical variables.
Implements One-Hot, Ordinal, and Target Encoding approaches.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from src.utils.logger import get_logger

logger = get_logger("preprocessing.encoding")


# =============================================================================
# ONE-HOT ENCODING
# =============================================================================

def encode_onehot(
    df: pd.DataFrame,
    cat_cols: list,
    drop_first: bool = True,
    encoder: OneHotEncoder = None,
    fit: bool = True
) -> tuple:
    """
    Apply One-Hot Encoding to categorical variables.
    """

    logger.info("Applying One-Hot Encoding")
    logger.info(f"  Columns: {cat_cols}")
    logger.info(f"  drop_first: {drop_first}")
    logger.info(f"  Mode: {'FIT' if fit else 'TRANSFORM'}")

    cat_cols = [col for col in cat_cols if col in df.columns]

    if not cat_cols:
        logger.warning(" No valid columns found for One-Hot Encoding")
        return df, encoder

    df = df.copy()

    if fit:
        encoder = OneHotEncoder(
            sparse_output=False,
            drop="first" if drop_first else None,
            handle_unknown="ignore"
        )
        encoded_array = encoder.fit_transform(df[cat_cols])
    else:
        if encoder is None:
            raise ValueError("Encoder must be provided when fit=False")
        encoded_array = encoder.transform(df[cat_cols])

    feature_names = encoder.get_feature_names_out(cat_cols)

    df_encoded = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df.index
    )

    df_result = df.drop(columns=cat_cols)
    df_result = pd.concat([df_result, df_encoded], axis=1)

    logger.info(f" One-Hot Encoding completed ({len(feature_names)} features)")
    return df_result, encoder


# =============================================================================
# ORDINAL ENCODING
# =============================================================================

def encode_ordinal(
    df: pd.DataFrame,
    cat_cols: list,
    categories_order: dict = None,
    encoder: OrdinalEncoder = None,
    fit: bool = True
) -> tuple:
    """
    Apply Ordinal Encoding to categorical variables.
    """

    logger.info("Applying Ordinal Encoding")
    logger.info(f"  Columns: {cat_cols}")
    logger.info(f"  Mode: {'FIT' if fit else 'TRANSFORM'}")

    if categories_order is None:
        categories_order = {}

    cat_cols = [col for col in cat_cols if col in df.columns]

    if not cat_cols:
        logger.warning(" No valid columns found for Ordinal Encoding")
        return df, encoder

    df = df.copy()

    if fit:
        categories = []

        for col in cat_cols:
            if col in categories_order:
                ordered_list = categories_order[col]
                logger.info(f"  {col}: custom order → {ordered_list}")
            else:
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

        df[cat_cols] = encoder.fit_transform(df[cat_cols])

    else:
        if encoder is None:
            raise ValueError("Encoder must be provided when fit=False")
        df[cat_cols] = encoder.transform(df[cat_cols])

    logger.info(" Ordinal Encoding completed")
    return df, encoder


# =============================================================================
# TARGET ENCODING (FIT ONLY)
# =============================================================================

def encode_target(
    df: pd.DataFrame,
    cat_cols: list,
    target_col: str,
    smoothing: float = 10.0
) -> pd.DataFrame:
    """
    Apply Target Encoding (Mean Encoding) to categorical variables.
    FIT ONLY (never applied on test).
    """

    logger.warning(" Target Encoding can cause leakage. FIT ONLY.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    cat_cols = [col for col in cat_cols if col in df.columns]

    if not cat_cols:
        logger.warning(" No valid columns found for Target Encoding")
        return df

    df = df.copy()
    global_mean = df[target_col].mean()

    for col in cat_cols:
        stats = df.groupby(col)[target_col].agg(["mean", "count"])

        stats["smoothed_mean"] = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )

        encoding_map = stats["smoothed_mean"].to_dict()
        df[col] = df[col].map(encoding_map).fillna(global_mean)

    logger.info(" Target Encoding completed")
    return df


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def encode_categorical(
    df: pd.DataFrame,
    onehot_cols: list = None,
    ordinal_cols: list = None,
    target_cols: dict = None,
    encoders: dict = None,
    fit: bool = True
) -> tuple:
    """
    Orchestrates all categorical encoding strategies.
    Supports fit / transform.
    """

    logger.info("=" * 80)
    logger.info("CATEGORICAL ENCODING ORCHESTRATION")
    logger.info(f"Mode: {'FIT' if fit else 'TRANSFORM'}")
    logger.info("=" * 80)

    if onehot_cols is None:
        onehot_cols = []
    if ordinal_cols is None:
        ordinal_cols = []
    if target_cols is None:
        target_cols = {}

    if fit:
        encoders = {}

    # One-Hot
    if onehot_cols:
        df, enc = encode_onehot(
            df,
            onehot_cols,
            encoder=None if fit else encoders.get("onehot"),
            fit=fit
        )
        encoders["onehot"] = enc

    # Ordinal
    if ordinal_cols:
        df, enc = encode_ordinal(
            df,
            ordinal_cols,
            encoder=None if fit else encoders.get("ordinal"),
            fit=fit
        )
        encoders["ordinal"] = enc

    # Target (FIT ONLY)
    if fit and target_cols and "columns" in target_cols and "target" in target_cols:
        df = encode_target(
            df,
            target_cols["columns"],
            target_cols["target"],
            smoothing=target_cols.get("smoothing", 10.0)
        )
        encoders["target"] = "applied"

    logger.info("=" * 80)
    logger.info("CATEGORICAL ENCODING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df, encoders
