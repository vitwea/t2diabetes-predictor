"""
UPDATED run_preprocessing.py
Integration with feature_selection.py

Pipeline:
1. [cleaning] Remove rows with missing glucose
2. [cleaning] Fix invalid BP measurements
3. [cleaning] Remove ID columns
4. [types] Define feature types
5. [clinical_rules] Reconstruct BMI
6. [imputation] KNN imputation for anthropometric
7. [imputation] Median imputation for numeric
8. [imputation] Mode imputation for categorical
9. [clinical_rules] Apply domain knowledge rules
10. [feature_engineering] Create engineered features
11. [encoding] Encode categorical variables
12. [feature_selection] Select top features
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import joblib

# =============================================================================
# Imports
# =============================================================================

from src.preprocessing.cleaning import (
    remove_glucose_nan,
    bpxdia_nan,
    remove_id_columns
)

from src.preprocessing.define_types import define_feature_types

from src.preprocessing.split import split_dataset

from src.features.clinical_rules import reconstruct_bmi, apply_clinical_rules
from src.preprocessing.imputation import (
    impute_numeric_knn,
    impute_numeric_median,
    impute_categorical_mode,
    check_missing_values
)
from src.preprocessing.encoding import encode_categorical
from src.features.feature_engineering import create_all_features
from src.features.feature_selection import select_top_features, save_features
from src.utils.logger import get_logger

# =============================================================================
# Logger
# =============================================================================

logger = get_logger("preprocessing_pipeline")

# =============================================================================
# Helpers: artifact management and validations
# =============================================================================

ARTIFACTS_PATH = Path("data/dataset/preprocessing_artifacts.pkl")
ARTIFACTS_JOBLIB_PATH = Path("data/dataset/preprocessing_artifacts.joblib")


def save_artifacts(artifacts: dict, path: Path = ARTIFACTS_JOBLIB_PATH):
    """Save artifacts to disk using joblib for sklearn-compatible objects.
    Comment: joblib is preferred for sklearn objects; fallback to pickle if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(artifacts, path)
        logger.info(f"Artifacts saved to {path}")
    except Exception:
        # Fallback to pickle if joblib fails for any reason
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(artifacts, f)
        logger.info(f"Artifacts saved to {path.with_suffix('.pkl')}")


def load_artifacts(path: Path = ARTIFACTS_JOBLIB_PATH) -> dict:
    """Load artifacts from disk if present. Returns empty dict if not found."""
    if path.exists():
        return joblib.load(path)
    pkl = path.with_suffix(".pkl")
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return {}


def require_artifact(artifacts: dict, key: str):
    """Raise a clear error if a required top-level artifact key is missing."""
    if key not in artifacts:
        raise RuntimeError(
            f"Required artifact '{key}' not found. Run preprocessing with mode='fit' first "
            "and persist artifacts before running transform."
        )


def ensure_columns_exist(df: pd.DataFrame, cols: list, fill_value=np.nan):
    """Ensure columns exist in df; if missing, create them with fill_value and log."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns in transform: {missing}. Creating with fill_value={fill_value}.")
        for c in missing:
            df[c] = fill_value
    return df


# =============================================================================
# POST-SPLIT PREPROCESSING (Steps 6–12)
# =============================================================================

def preprocessing(
    df: pd.DataFrame,
    encoding_config: dict,
    select_top: bool,
    mode: str,
    artifacts: dict,
    target_col: str = "diabetes_risk"
):
    """
    Preprocessing steps that MUST be applied after train/test split.
    mode: "fit" | "transform"
    Notes:
      - In 'fit' mode, functions should store fitted objects into artifacts.
      - In 'transform' mode, artifacts must contain the fitted objects.
    """

    # Basic mode validation
    if mode not in {"fit", "transform"}:
        raise ValueError("mode must be 'fit' or 'transform'")

    # If feature selection expects a target column during fit, ensure it's present
    if mode == "fit" and select_top:
        if target_col not in df.columns:
            raise RuntimeError(f"Target column '{target_col}' must be present in df during fit for feature selection.")


    # -------------------------------------------------------------------------
    # Step 6: Clinical rules
    # -------------------------------------------------------------------------
    df = apply_clinical_rules(df)

    # -------------------------------------------------------------------------
    # Step 7: Numerical imputation (KNN for anthropometric)
    # -------------------------------------------------------------------------
    knn_cols = ["waist_cm", "weight_kg", "height_cm", "bmi"]
    df = impute_numeric_knn(
        df,
        knn_cols,
        n_neighbors=5,
        mode=mode,
        artifacts=artifacts
    )

    # Validate that imputer artifact was stored when fitting (best-effort)
    if mode == "fit":
        # Comment: check the 'imputation' namespace for the 'knn' key (matches imputation functions)
        if "knn" not in artifacts.get("imputation", {}):
            logger.warning("KNN imputer not found in artifacts['imputation']. Ensure the function stores the imputer.")

    numeric_cols = [
        "creatinine", "bmi", "waist_cm", "weight_kg", "height_cm",
        "systolic_bp", "diastolic_bp", "hdl_cholesterol",
        "total_cholesterol", "triglycerides", "sleep_hours"
    ]
    df = impute_numeric_median(
        df,
        numeric_cols,
        mode=mode,
        artifacts=artifacts
    )

    if mode == "fit":
        # Comment: check the 'imputation' namespace for the 'median' key
        if "median" not in artifacts.get("imputation", {}):
            logger.warning("Median imputer not found in artifacts['imputation']. Ensure the function stores the medians.")

    # -------------------------------------------------------------------------
    # Step 8: Categorical imputation
    # -------------------------------------------------------------------------
    categorical_cols = [
        "gender", "ethnicity", "smoker", "hypertension",
        "liver_disease", "heart_disease", "income_poverty_ratio"
    ]
    df = impute_categorical_mode(
        df,
        categorical_cols,
        mode=mode,
        artifacts=artifacts
    )

    if mode == "fit":
        # Comment: check the 'imputation' namespace for the 'mode' key
        if "mode" not in artifacts.get("imputation", {}):
            logger.warning("Categorical mode imputer not found in artifacts['imputation']. Ensure the function stores the modes.")


    # -------------------------------------------------------------------------
    # Step 9: Feature engineering
    # -------------------------------------------------------------------------
    df = create_all_features(df)

    # -------------------------------------------------------------------------
    # Step 10: Missing values check
    # -------------------------------------------------------------------------
    check_missing_values(df, verbose=(mode == "fit"))

    # -------------------------------------------------------------------------
    # Step 11: Encoding
    # -------------------------------------------------------------------------
    if encoding_config:
        if mode == "fit":
            df, encoders = encode_categorical(
                df,
                onehot_cols=encoding_config.get("onehot_cols", []),
                ordinal_cols=encoding_config.get("ordinal_cols", []),
                target_cols=None
            )
            artifacts["encoders"] = encoders
            # Basic validation
            if artifacts.get("encoders") is None:
                logger.warning("encode_categorical returned no encoders. Check implementation.")
        else:
            # Validate encoders exist
            require_artifact(artifacts, "encoders")
            df, _ = encode_categorical(
                df,
                onehot_cols=encoding_config.get("onehot_cols", []),
                ordinal_cols=encoding_config.get("ordinal_cols", []),
                target_cols=None,
                encoders=artifacts["encoders"],
                fit=False
            )

    # -------------------------------------------------------------------------
    # Step 12: Feature selection
    # -------------------------------------------------------------------------
    if select_top:
        if mode == "fit":
            logger.info("Running feature selection (FIT)")
            # select_top_features is expected to accept df with target column included
            df_sel, selected_features = select_top_features(df)
            # If function returns df and features, use them; otherwise try alternative return
            if selected_features is None:
                # fallback: if select_top_features returned only features
                if isinstance(df_sel, list):
                    selected_features = df_sel
                    # keep df as-is
                    df_sel = df
                else:
                    raise RuntimeError("select_top_features did not return selected features. Check its signature.")
            df = df_sel

            artifacts["selected_features"] = selected_features
            # Save selected features to disk (human-readable)
            save_features(selected_features)

        else:
            logger.info("Applying feature selection from TRAIN (TRANSFORM)")
            require_artifact(artifacts, "selected_features")
            selected_features = artifacts["selected_features"]

            # If some selected features are missing in df, create them with NaN (so downstream models get consistent columns)
            missing = set(selected_features) - set(df.columns)
            if missing:
                logger.warning(f"Test data is missing selected features: {sorted(missing)}. Creating them with NaN.")
                for c in sorted(missing):
                    df[c] = np.nan

            # Enforce order and drop non-selected features
            df = df[selected_features]

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    logger.info("Loading dataset\n")
    df = pd.read_parquet("data/nhanes_data/cleaned/dataset_cleaned.parquet")

    # -------------------------------------------------------------------------
    # PRE-SPLIT STEPS (1–5)
    # -------------------------------------------------------------------------
    logger.info("Running PRE-SPLIT preprocessing (Steps 1–5)\n")

    df = remove_glucose_nan(df)
    df = bpxdia_nan(df)
    df = remove_id_columns(df, id_cols=["SEQN", "subject_id", "ID"])
    df = define_feature_types(df)
    df = reconstruct_bmi(df)

    # -------------------------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        target_col="diabetes_risk",
        test_size=0.2,
        random_state=42,
        stratify=True
    )

    # -------------------------------------------------------------------------
    # ENCODING CONFIG
    # -------------------------------------------------------------------------
    encoding_config = {
        "onehot_cols": [],
        "ordinal_cols": ["bmi_category", "bp_category", "age_group", "ethnicity", "hypertension"]
    }

    artifacts = {}

    # -------------------------------------------------------------------------
    # TRAIN → FIT
    # -------------------------------------------------------------------------
    logger.info("Running TRAIN preprocessing (FIT)\n")

    X_train_with_target = X_train.copy()
    X_train_with_target["diabetes_risk"] = y_train

    X_train_proc = preprocessing(
        X_train_with_target,
        encoding_config=encoding_config,
        select_top=True,
        mode="fit",
        artifacts=artifacts,
        target_col="diabetes_risk"
    )

    # -------------------------
    # Add metadata and lightweight summaries to artifacts (optional but recommended)
    # -------------------------
    # Comment: store feature types (dtypes) for reproducibility and schema checks
    artifacts.setdefault("feature_types", {})
    artifacts["feature_types"].update({col: str(dtype) for col, dtype in X_train_proc.dtypes.items()})

    # Comment: collect imputation info (which columns were imputed and parameters used)
    imputation = artifacts.get("imputation", {})
    imputation_info = {
        "knn_info": imputation.get("knn_info", {}),  # expected to be set by impute_numeric_knn
        "median_cols": list(imputation.get("median", {}).keys()) if isinstance(imputation.get("median", {}), dict) else [],
        "mode_cols": list(imputation.get("mode", {}).keys()) if isinstance(imputation.get("mode", {}), dict) else []
    }
    artifacts["imputation_info"] = imputation_info

    # Comment: encoding info - store config and a light summary of encoders if available
    encoding_info = {
        "encoding_config": encoding_config,
        "encoders_present": bool(artifacts.get("encoders")),
    }
    enc = artifacts.get("encoders")
    if enc and isinstance(enc, dict):
        # Comment: store only small metadata (e.g., categories_) to avoid serializing large objects twice
        encoding_info["encoder_summary"] = {k: getattr(v, "categories_", None) for k, v in enc.items()}
    artifacts["encoding_info"] = encoding_info

    # Comment: training stats and target distribution
    artifacts["training_stats"] = {
        "train_rows": int(X_train_proc.shape[0]),
        "train_cols": int(X_train_proc.shape[1]),
        "target_distribution": y_train.value_counts().to_dict() if y_train is not None else None
    }

    # Comment: lightweight data hash for traceability (hash of first 100 rows CSV)
    try:
        sample = X_train_proc.head(100).to_csv(index=False).encode("utf-8")
        artifacts["data_hash"] = sha256(sample).hexdigest()
    except Exception:
        artifacts["data_hash"] = None

    # Comment: meta information (versioning, timestamp). Optionally add git commit hash from CI.
    artifacts.setdefault("meta", {})
    artifacts["meta"].update({
        "pipeline_version": "v1.0",
        "created_at": datetime.utcnow().isoformat(),
        # "git_commit": "abc123"  # optional: fill from CI/CD environment
    })

    # Persist artifacts immediately after fit so they are available even if the script fails later
    save_artifacts(artifacts)

    # Drop target column before saving X_train
    if "diabetes_risk" in X_train_proc.columns:
        X_train_proc = X_train_proc.drop(columns=["diabetes_risk"])

    # -------------------------------------------------------------------------
    # TEST → TRANSFORM
    # -------------------------------------------------------------------------
    logger.info("Running TEST preprocessing (TRANSFORM)\n")

    # If running transform in a separate process, user should load artifacts from disk.
    # Here we assume same process; otherwise uncomment the following line to reload:
    # artifacts = load_artifacts()

    X_test_proc = preprocessing(
        X_test,
        encoding_config=encoding_config,
        select_top=True,
        mode="transform",
        artifacts=artifacts,
        target_col="diabetes_risk"
    )

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    logger.info("Saving outputs")

    out_dir = Path("data/dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_proc.to_parquet(out_dir / "X_train.parquet", index=False)
    X_test_proc.to_parquet(out_dir / "X_test.parquet", index=False)
    y_train.to_frame(name="diabetes_risk").to_parquet(out_dir / "y_train.parquet", index=False)
    y_test.to_frame(name="diabetes_risk").to_parquet(out_dir / "y_test.parquet", index=False)

    # Save artifacts again (final)
    save_artifacts(artifacts)

    logger.info("Pipeline completed successfully")