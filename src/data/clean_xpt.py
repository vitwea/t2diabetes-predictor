import pandas as pd
import pyreadstat
import os
import logging

# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------
# Variable mapping: raw NHANES → clean standardized names
# ---------------------------------------------------------
VARIABLES = {
    "SEQN": "id",
    "RIDAGEYR": "age_years",
    "RIAGENDR": "gender",
    "RIDRETH1": "ethnicity",
    "DIQ010": "diabetes_dx",
    "DIQ160": "family_history_diabetes",
    "LBXGLU": "glucose_mgdl",
    "LBXIN": "insulin_uUml",
    "LBXGH": "hba1c_percent",
    "BMXHT": "height_cm",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_cm",
    "BPXSY1": "sbp_1",
    "BPXSY2": "sbp_2",
    "BPXDI1": "dbp_1",
    "BPXDI2": "dbp_2",
    "DR1TPROT": "protein_g",
    "DR1TCARB": "carbs_g",
    "DR1TTFAT": "fat_g",
    "SMQ020": "smoking_status",
    "LBXTC": "chol_total_mgdl",
    "LBDHDL": "hdl_mgdl",
    "LBXTR": "triglycerides_mgdl",
    "LBDLDL": "ldl_mgdl",
    "LBXGGT": "ggt_iul",
    "LBXALT": "alt_iul",
    "LBXSCR": "creatinine_mgdl",
    "LBXUAPB": "urine_albumin_mgl",
    "LBXCRP": "crp_mgl",
}

# ---------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------
def clean_xpt(input_path, output_path):
    """
    Load an NHANES XPT file, select only relevant variables,
    rename them using standardized names, and export as Parquet.
    """

    logging.info(f"Processing file: {input_path}")

    try:
        # Load XPT file
        df, meta = pyreadstat.read_xport(input_path)
        logging.info(f"Loaded XPT file with {df.shape[0]} rows and {df.shape[1]} columns")

        # Select only variables of interest
        keep = [col for col in df.columns if col in VARIABLES]
        df = df[keep]
        logging.info(f"Selected {len(keep)} relevant variables")

        # Rename columns
        df = df.rename(columns=VARIABLES)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export to Parquet
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved cleaned file to: {output_path}")

    except Exception as e:
        logging.error(f"Error cleaning file {input_path}: {e}")