"""
Imputation strategies for numeric and categorical variables.
Separates KNN, median, and mode imputation logic into reusable functions.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.utils.logger import get_logger

logger = get_logger("preprocessing.imputation")


def impute_numeric_knn(
    df: pd.DataFrame,
    cols: list,
    n_neighbors: int,
    mode: str,
    artifacts: dict
) -> pd.DataFrame:
    """
    Apply KNN imputation to numeric columns and persist the fitted imputer in artifacts.
    In 'transform' mode the fitted imputer is required in artifacts["imputation"]["knn"].
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    # Ensure artifacts structure
    artifacts.setdefault("imputation", {})

    if mode == "fit":
        logger.info("Applying KNN imputation (FIT)")
        imputer = KNNImputer(n_neighbors=n_neighbors)
        # Fit-transform only the selected columns, preserve other columns
        df_loc = df[cols]
        df[cols] = imputer.fit_transform(df_loc)
        artifacts["imputation"]["knn"] = imputer
        logger.info("KNN imputer fitted and stored in artifacts['imputation']['knn']")
    else:
        logger.info("Applying KNN imputation (TRANSFORM)")
        if "knn" not in artifacts.get("imputation", {}):
            raise RuntimeError("KNN imputer not found in artifacts. Run preprocessing with mode='fit' first.")
        imputer = artifacts["imputation"]["knn"]
        df_loc = df[cols]
        df[cols] = imputer.transform(df_loc)
        logger.info("KNN imputer loaded from artifacts and applied to transform data")

    return df


def impute_numeric_median(
    df: pd.DataFrame,
    cols: list,
    mode: str,
    artifacts: dict
) -> pd.DataFrame:
    """
    Apply median imputation to numeric columns and persist medians in artifacts.
    In 'transform' mode medians are read from artifacts["imputation"]["median"].
    """
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    artifacts.setdefault("imputation", {})

    if mode == "fit":
        logger.info("Applying median imputation (FIT)")
        medians = {}
        for col in cols:
            median = df[col].median()
            medians[col] = median
            missing = int(df[col].isna().sum())
            if missing > 0:
                df[col] = df[col].fillna(median)
                logger.info(f"  {col}: {missing} values imputed with median ({median:.2f})")
        artifacts["imputation"]["median"] = medians
        logger.info("Median values stored in artifacts['imputation']['median']")
    else:
        logger.info("Applying median imputation (TRANSFORM)")
        if "median" not in artifacts.get("imputation", {}):
            raise RuntimeError("Median imputer not found in artifacts. Run preprocessing with mode='fit' first.")
        medians = artifacts["imputation"]["median"]
        for col, median in medians.items():
            if col in df.columns:
                missing_before = int(df[col].isna().sum())
                if missing_before > 0:
                    df[col] = df[col].fillna(median)
                    logger.info(f"  {col}: {missing_before} values imputed with median ({median:.2f})")
        logger.info("Median imputation applied using stored medians")

    return df


def impute_categorical_mode(
    df: pd.DataFrame,
    cols: list,
    mode: str,
    artifacts: dict
) -> pd.DataFrame:
    """
    Apply mode imputation to categorical columns and persist modes in artifacts.
    In 'transform' mode modes are read from artifacts["imputation"]["mode"].
    """
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    artifacts.setdefault("imputation", {})

    if mode == "fit":
        logger.info("Applying mode imputation (FIT)")
        modes = {}
        for col in cols:
            if df[col].isna().any():
                # guard against empty series for mode()
                mode_series = df[col].mode(dropna=True)
                if len(mode_series) == 0:
                    logger.warning(f"  {col}: no non-null values to compute mode; skipping")
                    continue
                mode_val = mode_series.iloc[0]
                modes[col] = mode_val
                missing = int(df[col].isna().sum())
                df[col] = df[col].fillna(mode_val)
                logger.info(f"  {col}: {missing} values imputed with mode ({mode_val})")
        artifacts["imputation"]["mode"] = modes
        logger.info("Categorical modes stored in artifacts['imputation']['mode']")
    else:
        logger.info("Applying mode imputation (TRANSFORM)")
        if "mode" not in artifacts.get("imputation", {}):
            raise RuntimeError("Categorical mode imputer not found in artifacts. Run preprocessing with mode='fit' first.")
        modes = artifacts["imputation"]["mode"]
        for col, mode_val in modes.items():
            if col in df.columns:
                missing_before = int(df[col].isna().sum())
                if missing_before > 0:
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"  {col}: {missing_before} values imputed with mode ({mode_val})")
        logger.info("Categorical mode imputation applied using stored modes")

    return df


def check_missing_values(df: pd.DataFrame, verbose: bool = True) -> dict:
    logger.info("Checking missing values")
    nan_summary = df.isnull().sum()
    total_missing = int(nan_summary.sum())
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0.0

    if verbose:
        logger.info(f"Total missing: {total_missing:,} ({missing_pct:.2f}%)\n")

    if total_missing > 0:
        cols_with_missing = nan_summary[nan_summary > 0].sort_values(ascending=False)
        if verbose:
            logger.info(f"Columns with missing values ({len(cols_with_missing)}):\n")
            for col, count in cols_with_missing.items():
                pct = (count / len(df)) * 100 if len(df) > 0 else 0.0
                logger.info(f"{col}: {count:,} ({pct:.2f}%)\n")
    else:
        logger.info("No missing values found\n")

    return {"total_missing": total_missing, "missing_pct": missing_pct}