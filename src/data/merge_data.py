"""
NHANES Data Merging - Binary Target Variable for Diabetes Risk
"""

import pandas as pd
import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("merge_data")

def normalize_columns_for_cycle(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """
    Rename columns to readable names.
    Multiple original names → ONE single readable name within cycle.
    """
    rename_dict = {}

    # Dynamic variables
    if "LBXTR" in df.columns:
        rename_dict["LBXTR"] = "triglycerides"
    if "LBXTLG" in df.columns:
        rename_dict["LBXTLG"] = "triglycerides"
    if "LBXTGL" in df.columns:
        rename_dict["LBXTGL"] = "triglycerides"
    if "LBXTC" in df.columns:
        rename_dict["LBXTC"] = "total_cholesterol"
    if "SLD010H" in df.columns:
        rename_dict["SLD010H"] = "sleep_hours"
    if "SLD012" in df.columns:
        rename_dict["SLD012"] = "sleep_hours"
    if "BPXSY1" in df.columns:
        rename_dict["BPXSY1"] = "systolic_bp"
    if "BPXOSY1" in df.columns:
        rename_dict["BPXOSY1"] = "systolic_bp"
    if "BPXDI1" in df.columns:
        rename_dict["BPXDI1"] = "diastolic_bp"
    if "BPXODI1" in df.columns:
        rename_dict["BPXODI1"] = "diastolic_bp"

    # Demographic variables
    if "SEQN" in df.columns:
        rename_dict["SEQN"] = "subject_id"
    if "RIDAGEYR" in df.columns:
        rename_dict["RIDAGEYR"] = "age_years"
    if "RIAGENDR" in df.columns:
        rename_dict["RIAGENDR"] = "gender"
    if "RIDRETH1" in df.columns:
        rename_dict["RIDRETH1"] = "ethnicity"
    if "INDFMPIR" in df.columns:
        rename_dict["INDFMPIR"] = "income_poverty_ratio"

    # Anthropometric variables
    if "BMXBMI" in df.columns:
        rename_dict["BMXBMI"] = "bmi"
    if "BMXHT" in df.columns:
        rename_dict["BMXHT"] = "height_cm"
    if "BMXWAIST" in df.columns:
        rename_dict["BMXWAIST"] = "waist_cm"
    if "BMXWT" in df.columns:
        rename_dict["BMXWT"] = "weight_kg"

    # Cardiovascular variables
    if "BPQ020" in df.columns:
        rename_dict["BPQ020"] = "hypertension"
    if "LBDHDD" in df.columns:
        rename_dict["LBDHDD"] = "hdl_cholesterol"

    # Glucose (KEEP ORIGINAL)
    if "LBXGH" in df.columns:
        rename_dict["LBXGH"] = "glucose_value"

    # Other variables
    if "LBXSCR" in df.columns:
        rename_dict["LBXSCR"] = "creatinine"
    if "MCQ160E" in df.columns:
        rename_dict["MCQ160E"] = "heart_disease"
    if "MCQ160L" in df.columns:
        rename_dict["MCQ160L"] = "liver_disease"
    if "SMQ020" in df.columns:
        rename_dict["SMQ020"] = "smoker"

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df

def merge_cycle_data(cycle_dir: Path, cycle: str) -> pd.DataFrame:
    """
    Merge all parquet files from ONE CYCLE by subject_id.
    """
    logger.info(f"\nProcessing cycle: {cycle}")
    parquet_files = sorted(cycle_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f" No parquet files found in {cycle_dir}")
        return None

    df_merged = None

    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            logger.info(f" Loaded {parquet_file.name}: {df.shape}")
            df = normalize_columns_for_cycle(df, cycle)

            if df_merged is None:
                df_merged = df
            else:
                logger.info(f" Merging with previous data...")
                df_merged = df_merged.merge(
                    df,
                    on="subject_id",
                    how="outer",
                    suffixes=("", "_new")
                )

        except Exception as e:
            logger.error(f" Error loading {parquet_file.name}: {e}")
            continue

    if df_merged is not None:
        logger.info(f"Cycle {cycle} final shape: {df_merged.shape}")
        return df_merged

    return None

def merge_all_cycles(
    cleaned_data_dir: str = "./data/nhanes_data/cleaned",
    output_file: str = "./data/nhanes_data/NHANES_consolidated.parquet"
) -> pd.DataFrame:
    """
    Merge all NHANES cycles and create binary target variable for diabetes risk.
    """
    logger.info("="*80)
    logger.info("MERGING ALL NHANES CYCLES")
    logger.info("="*80)

    cleaned_path = Path(cleaned_data_dir)
    cycle_dirs = sorted([d for d in cleaned_path.iterdir() if d.is_dir()])

    if not cycle_dirs:
        logger.error(f"No cycle directories found in {cleaned_data_dir}")
        return None

    logger.info(f"Found {len(cycle_dirs)} cycles\n")

    # Merge each cycle
    dfs_cycles = []

    for cycle_dir in cycle_dirs:
        cycle = cycle_dir.name
        df_cycle = merge_cycle_data(cycle_dir, cycle)

        if df_cycle is not None:
            dfs_cycles.append(df_cycle)

    if not dfs_cycles:
        logger.error("No valid data to merge")
        return None

    # Concatenate cycles vertically
    logger.info(f"\nCombining {len(dfs_cycles)} cycles...")
    df_consolidated = pd.concat(dfs_cycles, ignore_index=True, sort=False)

    logger.info(f"\nConsolidated dataset shape: {df_consolidated.shape}")
    logger.info(f" Total rows: {df_consolidated.shape[0]:,}")
    logger.info(f" Total columns: {df_consolidated.shape[1]}")

    # Show glucose_value statistics and create binary target variable
    if "glucose_value" in df_consolidated.columns:
        logger.info(f"\n✓ glucose_value (ORIGINAL):")
        logger.info(f" Min: {df_consolidated['glucose_value'].min():.2f}%")
        logger.info(f" Max: {df_consolidated['glucose_value'].max():.2f}%")
        logger.info(f" Mean: {df_consolidated['glucose_value'].mean():.2f}%")
        logger.info(f" Std: {df_consolidated['glucose_value'].std():.2f}%")
        logger.info(f" Missing: {df_consolidated['glucose_value'].isnull().sum():,} values")
        
        # ✅ CREATE BINARY TARGET VARIABLE
        # 0 = Normal (< 5.7% HbA1c)
        # 1 = Diabetes Risk (>= 5.7% HbA1c)
        df_consolidated['diabetes_risk'] = (df_consolidated['glucose_value'] >= 5.7).astype(int)
        
        # Log distribution of binary target variable
        logger.info(f"\n✓ diabetes_risk (BINARY TARGET - NEW):")
        logger.info(f" 0 (Normal, < 5.7%): {(df_consolidated['diabetes_risk'] == 0).sum():,} samples")
        logger.info(f" 1 (Diabetes Risk, >= 5.7%): {(df_consolidated['diabetes_risk'] == 1).sum():,} samples")
        logger.info(f" Missing: {df_consolidated['diabetes_risk'].isnull().sum():,} values")
        
        # Calculate percentage distribution
        class_dist = df_consolidated['diabetes_risk'].value_counts(normalize=True) * 100
        logger.info(f" Distribution: {class_dist.get(0, 0):.1f}% Normal | {class_dist.get(1, 0):.1f}% Risk")

    # Save
    logger.info(f"\nSaving file...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_consolidated.to_parquet(output_file, index=False, compression='snappy')

    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    logger.info(f"Consolidated file saved: {os.path.abspath(output_file)}")
    logger.info(f" Size: {file_size_mb:.2f} MB")
    logger.info("="*80)

    return df_consolidated

if __name__ == "__main__":
    df = merge_all_cycles()
    
    # Verificación
    if df is not None:
        logger.info("\n🔍 VERIFICACIÓN FINAL:")
        logger.info(f"Columnas disponibles: {df.columns.tolist()}")
        logger.info(f"\nDistribución de diabetes_risk:")
        logger.info(f"\n{df['diabetes_risk'].value_counts()}")
        logger.info(f"\nPrimeras 5 filas (muestra):")
        logger.info(f"\n{df[['subject_id', 'glucose_value', 'diabetes_risk']].head()}")