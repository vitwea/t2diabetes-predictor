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
    # Demographics
    "SEQN": "id",
    "RIDAGEYR": "age_years",
    "RIAGENDR": "gender",
    "RIDRETH1": "ethnicity",
    
    # Diabetes
    "DIQ010": "diabetes_dx",
    "DIQ160": "family_history_diabetes",
    
    # Glucose metabolism
    "LBXGLU": "glucose_mgdl",
    "LBXGH": "hba1c_percent",
    "LBXIN": "insulin_uUml",
    
    # Anthropometry
    "BMXHT": "height_cm",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_cm",
    
    # Blood Pressure
    "BPXSY1": "sbp_1",
    "BPXSY2": "sbp_2",
    "BPXDI1": "dbp_1",
    "BPXDI2": "dbp_2",
    
    # Lipids
    "LBXTC": "chol_total_mgdl",
    "LBDHDL": "hdl_mgdl",
    "LBXTR": "triglycerides_mgdl",
    
    # Liver function
    "LBXGGT": "ggt_iul",
    "LBXALT": "alt_iul",
    
    # Kidney function
    "LBXSCR": "creatinine_mgdl",
    "LBXUAPB": "urine_albumin_cr",
    
    # Inflammation
    "LBXCRP": "crp_mgl",  # Only 2015-2016+
    
    # Smoking
    "SMQ020": "smoking_status",
    
    # Dietary
    "DR1TPROT": "protein_g",
    "DR1TCARB": "carbs_g",
    "DR1TTFAT": "fat_g",
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