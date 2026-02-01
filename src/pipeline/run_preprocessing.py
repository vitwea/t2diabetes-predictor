"""
UPDATED run_preprocessing.py
Integración con feature_selection.py

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

import pandas as pd
import pickle
import logging

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


# =============================================================================
# Logger
# =============================================================================

logger = logging.getLogger("preprocessing")


# =============================================================================
# POST-SPLIT PREPROCESSING (Steps 6–12)
# =============================================================================

def preprocessing(
    df: pd.DataFrame,
    encoding_config: dict,
    select_top: bool,
    mode: str,
    artifacts: dict
):
    """
    Preprocessing steps that MUST be applied after train/test split.
    mode: "fit" | "transform"
    """

    # -------------------------------------------------------------------------
    # Step 6: Numerical imputation
    # -------------------------------------------------------------------------
    knn_cols = ["waist_cm", "weight_kg", "height_cm", "bmi"]
    df = impute_numeric_knn(df, knn_cols, n_neighbors=5)

    numeric_cols = [
        "creatinine", "bmi", "waist_cm", "weight_kg", "height_cm",
        "systolic_bp", "diastolic_bp", "hdl_cholesterol",
        "total_cholesterol", "triglycerides", "sleep_hours"
    ]
    df = impute_numeric_median(df, numeric_cols, skip_cols=knn_cols)

    # -------------------------------------------------------------------------
    # Step 7: Categorical imputation
    # -------------------------------------------------------------------------
    categorical_cols = [
        "gender", "ethnicity", "smoker", "hypertension",
        "liver_disease", "heart_disease", "income_poverty_ratio"
    ]
    df = impute_categorical_mode(df, categorical_cols)

    # -------------------------------------------------------------------------
    # Step 8: Clinical rules
    # -------------------------------------------------------------------------
    df = apply_clinical_rules(df)

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
        else:
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
            df, selected_features = select_top_features(df)
            artifacts["selected_features"] = selected_features
            save_features(selected_features)
        else:
            df = df[artifacts["selected_features"]]

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    logger.info("Loading dataset")
    df = pd.read_parquet("data/nhanes_data/cleaned/dataset_cleaned.parquet")

    # -------------------------------------------------------------------------
    # PRE-SPLIT STEPS (1–5)
    # -------------------------------------------------------------------------
    logger.info("Running PRE-SPLIT preprocessing (Steps 1–5)")

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
        "ordinal_cols": ["bmi_category", "bp_category", "age_group", "ethnicity"]
    }

    artifacts = {}

    # -------------------------------------------------------------------------
    # TRAIN → FIT
    # -------------------------------------------------------------------------
    logger.info("Running TRAIN preprocessing (FIT)")

    X_train_with_target = X_train.copy()
    X_train_with_target["diabetes_risk"] = y_train

    X_train_proc = preprocessing(
        X_train_with_target,
        encoding_config=encoding_config,
        select_top=True,
        mode="fit",
        artifacts=artifacts
    )
    X_train_proc = X_train_proc.drop(columns=["diabetes_risk"])
    # -------------------------------------------------------------------------
    # TEST → TRANSFORM
    # -------------------------------------------------------------------------
    logger.info("Running TEST preprocessing (TRANSFORM)")
    X_test_proc = preprocessing(
        X_test,
        encoding_config=encoding_config,
        select_top=True,
        mode="transform",
        artifacts=artifacts
    )

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    logger.info("Saving outputs")

    X_train_proc.to_parquet("data/dataset/X_train.parquet", index=False)
    X_test_proc.to_parquet("data/dataset/X_test.parquet", index=False)
    y_train.to_frame(name="diabetes_risk").to_parquet("data/dataset/y_train.parquet",index=False)
    y_test.to_frame(name="diabetes_risk").to_parquet("data/dataset/y_test.parquet",index=False)

    with open("data/dataset/preprocessing_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    logger.info("Pipeline completed successfully")