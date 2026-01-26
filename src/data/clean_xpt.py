"""
NHANES XPT to Parquet Converter with Variable Normalization
===========================================================

This script processes National Health and Nutrition Examination Survey (NHANES) 
XPT data files, extracts target health variables, normalizes column names across 
different survey cycles, and exports to optimized Parquet format.

Features:
- Robust XPT file loading with fallback strategies
- Automatic variable name normalization (handles NHANES naming changes)
- Conversion to numeric types with missing value handling
- Consolidation of multiple survey cycles
- Detailed logging and normalization reports

Author: Data Pipeline
Date: 2026
"""

import pandas as pd
import pyreadstat
import os
from pathlib import Path
from src.utils.logger import get_logger

# ---------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------
logger = get_logger("clean_xpt")

# ---------------------------------------------------------
# BASE TARGET VARIABLES (constant across cycles)
# ---------------------------------------------------------
BASE_TARGET_VARIABLES = {
    # Demographics: Age, Gender, Ethnicity, Family Poverty Index
    "DEMO": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"],
    
    # Body Measurements: BMI, Waist Circumference, Weight, Height
    "BMX": ["BMXBMI", "BMXWAIST", "BMXWT", "BMXHT"],
    
    # Blood Pressure Questionnaire
    "BPQ": ["BPQ020"],
    
    # HDL Cholesterol (Laboratory)
    "HDL": ["LBDHDD"],
    
    # Total Cholesterol (Laboratory)
    "TCHOL": ["LBXTC"],
    
    # Serum Creatinine (Kidney Function - Biomarkers/Proteins)
    "BIOPRO": ["LBXSCR"],
    
    # Medical Conditions Questionnaire: Cancer, Asthma
    "MCQ": ["MCQ160L", "MCQ160E"],
    
    # Smoking: Current Smoking Status
    "SMQ": ["SMQ020"],
    
    # Alcohol Use: Ever consumed alcohol
    "ALQ": ["ALQ101"],
    
    # Glucose (Fasting - Laboratory): Critical for diabetes analysis
    "GHB": ["LBXGH"],
}

# ---------------------------------------------------------
# VARIABLE NAME NORMALIZATION MAPPING
# ==================================================
# Maps old/new NHANES variable names to standardized names
# 
# NHANES periodically updates variable names and codes.
# This mapping ensures consistency across survey cycles:
# - 2011-2014 vs 2015+ might use different codes
# - 2011-2016 vs 2017+ protocol changes
# 
# Purpose: All survey cycles will have identical column names
# Example: LBXTR (old) and LBXTLG (new) both become TRIGLY
# ---------------------------------------------------------
VARIABLE_NORMALIZATION_MAP = {
    # TRIGLY - Triglycerides (lipid panel)
    "LBXTR": "TRIGLY",      # Old name (2011-2020)
    "LBXTLG": "TRIGLY",     # New name (2021+)
    
    # SLQ - Sleep Quality Questions
    "SLD010H": "SLQ",       # Old name (2011-2014)
    "SLD012": "SLQ",        # New name (2015+)
    
    # BPX_SYS - Systolic Blood Pressure (first measurement)
    "BPXSY1": "BPX_SYS",    # Old name (2011-2016)
    "BPXOSY1": "BPX_SYS",   # New name with 'O' prefix (2017+)
    
    # BPX_DIA - Diastolic Blood Pressure (first measurement)
    "BPXDI1": "BPX_DIA",    # Old name (2011-2016)
    "BPXODI1": "BPX_DIA",   # New name with 'O' prefix (2017+)
}

# ---------------------------------------------------------
# FUNCTION: Get Target Variables for Specific Cycle
# ---------------------------------------------------------
def get_target_variables_for_cycle(cycle):
    """
    Returns TARGET_VARIABLES adjusted for specific NHANES survey cycle.
    
    NHANES changed variable names in different periods due to:
    - Protocol updates
    - Equipment changes
    - Questionnaire revisions
    
    Variables by cycle:
    - TRIGLY: 2011-2020 → LBXTR, 2021+ → LBXTGL
    - SLQ: 2011-2014 → SLD010H, 2015+ → SLD012
    - BPX: 2011-2016 → [BPXSY1, BPXDI1], 2017+ → [BPXOSY1, BPXODI1]
    
    Args:
        cycle (str): Cycle identifier (e.g., "2011-2012", "2021-2022")
    
    Returns:
        dict: TARGET_VARIABLES updated with correct variable names for cycle
    """
    target_vars = BASE_TARGET_VARIABLES.copy()
    
    # Extract year from cycle string (first year of range)
    try:
        year = int(cycle.split("-")[0])
    except:
        year = 2020  # Default fallback

    # TRIGLY: Variable name changed in 2021
    if year >= 2021:
        target_vars["TRIGLY"] = ["LBXTLG"]  # 2021 onwards
    else:
        target_vars["TRIGLY"] = ["LBXTR"]   # 2011-2020

    # SLQ: Sleep questionnaire changed in 2015
    if year >= 2015:
        target_vars["SLQ"] = ["SLD012"]     # 2015+
    else:
        target_vars["SLQ"] = ["SLD010H"]    # 2011-2014

    # BPX: Blood pressure protocol updated in 2017 (added 'O' prefix)
    if year >= 2017:
        target_vars["BPX"] = ["BPXOSY1", "BPXODI1"]  # 2017+
    else:
        target_vars["BPX"] = ["BPXSY1", "BPXDI1"]    # 2011-2016

    return target_vars


# ---------------------------------------------------------
# FUNCTION: Create Flattened Variable Dictionary
# ---------------------------------------------------------
def get_all_variables_for_cycle(cycle):
    """
    Creates a flat dictionary of ALL variables for a specific cycle.
    
    Converts nested dictionary structure to flat key-value pairs
    for easier lookup during data extraction.
    
    Args:
        cycle (str): NHANES cycle identifier
    
    Returns:
        dict: Flattened dictionary where keys and values are variable names
    """
    target_vars = get_target_variables_for_cycle(cycle)
    all_vars = {}
    
    # Flatten: iterate through categories and extract variable names
    for filecode, variables in target_vars.items():
        for var in variables:
            all_vars[var] = var
    
    return all_vars


# ---------------------------------------------------------
# FUNCTION: Normalize Column Names
# ---------------------------------------------------------
def normalize_column_names(df):
    """
    Normalizes DataFrame column names using VARIABLE_NORMALIZATION_MAP.
    
    Renames variables with old/new NHANES codes to standardized names.
    This ensures consistency across survey cycles.
    
    Transformation examples:
    - LBXTR (2011-2020) → TRIGLY
    - LBXTLG (2021+) → TRIGLY (same normalized name)
    - BPXSY1 (2011-2016) → BPX_SYS
    - BPXOSY1 (2017+) → BPX_SYS (same normalized name)
    
    Args:
        df (pd.DataFrame): DataFrame with original NHANES column names
    
    Returns:
        pd.DataFrame: DataFrame with normalized column names
    """
    # Build rename dictionary: only include columns that exist in DataFrame
    rename_dict = {}
    
    for old_name, normalized_name in VARIABLE_NORMALIZATION_MAP.items():
        if old_name in df.columns:
            rename_dict[old_name] = normalized_name
    
    # Apply renaming operation
    df_normalized = df.rename(columns=rename_dict)
    
    # Log the normalization operation
    if rename_dict:
        logger.info(f"Normalized {len(rename_dict)} columns: {rename_dict}")
    
    return df_normalized


# ---------------------------------------------------------
# FUNCTION: Load XPT File with Robust Fallback Strategy
# ---------------------------------------------------------
def load_xpt_robust(input_path):
    """
    Loads XPT (SAS export format) file with multiple fallback strategies.
    
    XPT files can have compatibility issues. This function attempts
    three different loading methods in order of preference:
    1. pyreadstat (preferred: fast and reliable)
    2. pandas.read_sas (alternative: more compatible)
    3. pyreadstat with latin1 encoding (fallback: encoding fix)
    
    Args:
        input_path (str): Path to XPT file
    
    Returns:
        pd.DataFrame: Loaded DataFrame, or None if all methods fail
    """
    try:
        # Attempt 1: pyreadstat (fastest and most reliable)
        try:
            df, meta = pyreadstat.read_xport(input_path)
            logger.info(f"[pyreadstat] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        except Exception as e1:
            logger.warning(f"[pyreadstat] Failed: {str(e1)[:60]}. Trying pandas.read_sas...")
            
            # Attempt 2: pandas.read_sas (more compatible format)
            try:
                df = pd.read_sas(input_path, format="xport")
                logger.info(f"[pandas.read_sas] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                return df
            except Exception as e2:
                logger.warning(f"[pandas.read_sas] Failed: {str(e2)[:60]}. Trying alternative encoding...")
                
                # Attempt 3: pyreadstat with alternative encoding
                try:
                    df, meta = pyreadstat.read_xport(input_path, encoding='latin1')
                    logger.info(f"[pyreadstat latin1] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                    return df
                except Exception as e3:
                    logger.error(f"All readers failed")
                    return None
    
    except Exception as e:
        logger.error(f"Unexpected error loading XPT: {e}")
        return None


# ---------------------------------------------------------
# FUNCTION: Clean XPT File and Export as Parquet
# ---------------------------------------------------------
def clean_xpt(input_path, output_path, cycle=None):
    """
    Main cleaning pipeline: loads XPT, extracts variables, 
    normalizes names, and exports as Parquet.
    
    Pipeline steps:
    1. Load XPT file with robust fallback strategy
    2. Verify SEQN (individual identifier) exists
    3. Get target variables for specific NHANES cycle
    4. Select only target columns
    5. Convert to numeric type (handle missing values)
    6. Normalize variable names for consistency
    7. Export to Parquet format with Snappy compression
    
    Args:
        input_path (str): Path to input XPT file
        output_path (str): Path to output Parquet file
        cycle (str): NHANES cycle identifier (e.g., "2011-2012")
                    REQUIRED - used to select correct variable names
    
    Returns:
        bool: True if successful, False if processing failed
    """
    logger.info(f"Processing: {os.path.basename(input_path)}")
    
    # Step 1: Load XPT with robust fallback mechanisms
    df = load_xpt_robust(input_path)
    if df is None or df.shape[0] == 0:
        logger.warning(f"Skipped empty or corrupted file: {input_path}")
        return False
    
    # Step 2: Verify SEQN (individual ID) exists - ALWAYS required
    if "SEQN" not in df.columns:
        logger.error(f"SEQN not found in {input_path}. Skipping.")
        return False
    
    # Step 3: Verify cycle parameter provided (needed for variable selection)
    if not cycle:
        logger.error(f"Cycle not provided for {input_path}. Skipping.")
        return False
    
    # Step 4: Get all target variables for this specific cycle
    all_variables = get_all_variables_for_cycle(cycle)
    logger.info(f"Target variables count: {len(all_variables)}")
    
    # Step 5: Select only target columns that exist + SEQN identifier
    target_cols = ["SEQN"] + [var for var in all_variables.keys() if var in df.columns]
    df_filtered = df[target_cols].copy()
    
    variables_found = len(target_cols) - 1  # Exclude SEQN from count
    logger.info(f"Found {variables_found}/{len(all_variables)} target variables")
    
    if variables_found == 0:
        logger.warning(f"No target variables found in {input_path}. Skipping.")
        return False
    
    # Step 6: Convert all numeric columns (handle missing values)
    for col in df_filtered.columns:
        if col != "SEQN":
            # pd.to_numeric converts values; non-numeric become NaN
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Step 7: ⭐ NORMALIZE VARIABLE NAMES FOR CONSISTENCY
    df_normalized = normalize_column_names(df_filtered)
    
    # Step 8: Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Step 9: Export to Parquet (compressed with Snappy for storage efficiency)
    df_normalized.to_parquet(output_path, index=False, compression='snappy')
    logger.info(f"✓ Saved: {os.path.basename(output_path)} | {df_normalized.shape[0]:,} rows × {df_normalized.shape[1]} cols")
    
    return True


# ---------------------------------------------------------
# FUNCTION: Main Processing Pipeline
# ---------------------------------------------------------
def main():
    """
    Main execution function: processes all downloaded NHANES XPT files.
    
    Workflow:
    1. Initialize logging and validate raw data directory
    2. Discover all XPT files in raw data directory
    3. Group files by NHANES survey cycle
    4. Process each cycle independently
    5. Apply cycle-specific variable selection
    6. Normalize variable names and export to Parquet
    7. Log summary statistics
    
    Output:
    - ./data/nhanes_data/cleaned/
        ├── 2011-2012/
        │   ├── DEMO_B.parquet
        │   ├── BMX_B.parquet
        │   └── ...
        ├── 2013-2014/
        │   └── ...
        └── 2021-2022/
            └── ...
    """
    logger.info("="*80)
    logger.info("NHANES XPT CLEANING PIPELINE - WITH VARIABLE NORMALIZATION")
    logger.info("="*80)
    
    # Configure data paths
    RAW_DATA_DIR = "./data/nhanes_data/raw"
    CLEANED_DATA_DIR = "./data/nhanes_data/cleaned"
    
    # Verify raw data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        logger.error("Please run download_data.py first")
        return
    
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    
    # Discover all XPT files in raw data directory
    xpt_files = list(Path(RAW_DATA_DIR).rglob("*.xpt"))
    logger.info(f"Found {len(xpt_files)} XPT files to process")
    
    if len(xpt_files) == 0:
        logger.warning("No XPT files found. Run download_data.py first.")
        return
    
    # Group files by NHANES survey cycle
    cycles = {}
    for xpt_file in xpt_files:
        # Extract cycle from directory name (parent folder)
        cycle = xpt_file.parent.name
        if cycle not in cycles:
            cycles[cycle] = []
        cycles[cycle].append(xpt_file)
    
    # Process each cycle
    total_processed = 0
    total_failed = 0
    
    for cycle_idx, (cycle, files) in enumerate(sorted(cycles.items()), 1):
        logger.info(f"\n[Cycle {cycle_idx}/{len(cycles)}] Processing {cycle}")
        cycle_output_dir = os.path.join(CLEANED_DATA_DIR, cycle)
        os.makedirs(cycle_output_dir, exist_ok=True)
        
        # Process each file in the cycle
        for file_idx, xpt_file in enumerate(sorted(files), 1):
            output_file = os.path.join(
                cycle_output_dir,
                xpt_file.stem + ".parquet"
            )
            
            # Process XPT file with cycle-specific variable selection
            success = clean_xpt(str(xpt_file), output_file, cycle=cycle)
            
            if success:
                total_processed += 1
            else:
                total_failed += 1
    
    # Log summary statistics
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED")
    logger.info(f"✓ Successfully processed: {total_processed} files")
    logger.info(f"✗ Failed: {total_failed} files")
    logger.info(f"Output directory: {os.path.abspath(CLEANED_DATA_DIR)}")
    logger.info("="*80)


# ---------------------------------------------------------
# FUNCTION: Consolidate All Cycles into Single Dataset
# ---------------------------------------------------------
def merge_all_cycles():
    """
    Consolidates Parquet files from all NHANES cycles into one dataset.
    
    Operations:
    1. Load all cycle-specific Parquet files
    2. Add CYCLE column to track data source
    3. Concatenate into single DataFrame
    4. Create SEQN_UNIQUE identifier (SEQN + Cycle)
    5. Export consolidated file
    
    Benefits of consolidation:
    - Single file for analysis across all cycles
    - Normalized variable names (already applied)
    - Traceable data source via CYCLE column
    - Optimized compression format (Snappy)
    
    Output:
    - ./data/nhanes_data/NHANES_consolidated.parquet
    """
    logger.info("\n" + "="*80)
    logger.info("MERGING ALL CYCLES INTO SINGLE DATASET (NORMALIZED)")
    logger.info("="*80)
    
    CLEANED_DATA_DIR = "./data/nhanes_data/cleaned"
    OUTPUT_FILE = "./data/nhanes_data/NHANES_consolidated.parquet"
    
    # Discover all Parquet files from all cycles
    parquet_files = list(Path(CLEANED_DATA_DIR).rglob("*.parquet"))
    
    if len(parquet_files) == 0:
        logger.error("No parquet files found. Run main() first.")
        return
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Load and consolidate files
    dfs = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            
            # Add cycle identifier column (track data source)
            cycle = parquet_file.parent.name
            df['CYCLE'] = cycle
            
            dfs.append(df)
            logger.info(f"✓ Loaded {parquet_file.name}")
        except Exception as e:
            logger.error(f"✗ Failed to load {parquet_file.name}: {e}")
    
    if len(dfs) == 0:
        logger.error("No valid parquet files to merge")
        return
    
    # Concatenate all DataFrames
    df_consolidated = pd.concat(dfs, ignore_index=True)
    
    # Create unique identifier: SEQN varies by cycle, so combine both
    # Example: "12345_2011-2012" (individual 12345 from 2011-2012 cycle)
    df_consolidated['SEQN_UNIQUE'] = df_consolidated['SEQN'].astype(str) + '_' + df_consolidated['CYCLE']
    
    # Log dataset statistics
    logger.info(f"\nConsolidated dataset shape: {df_consolidated.shape}")
    logger.info(f"Total rows: {df_consolidated.shape[0]:,}")
    logger.info(f"Total columns: {df_consolidated.shape[1]}")
    logger.info(f"Columns: {', '.join(df_consolidated.columns.tolist())}")
    
    # Save consolidated file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_consolidated.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    logger.info(f"\n✓ Consolidated file saved: {os.path.abspath(OUTPUT_FILE)}")
    logger.info(f"Size: {file_size_mb:.2f} MB")
    logger.info("="*80)
    
    return df_consolidated


# ---------------------------------------------------------
# FUNCTION: Generate Normalization Report
# ---------------------------------------------------------
def generate_normalization_report():
    """
    Generates detailed report of variable normalization across cycles.
    
    Report includes:
    1. All normalized variables and their distributions
    2. Row count per variable and cycle
    3. List of all normalization mappings applied
    
    Purpose:
    - Verify normalization was applied correctly
    - Track variable availability by cycle
    - Document data lineage for reproducibility
    
    Output:
    - Logged to console/file with detailed statistics
    """
    logger.info("\n" + "="*80)
    logger.info("VARIABLE NORMALIZATION REPORT")
    logger.info("="*80)
    
    CLEANED_DATA_DIR = "./data/nhanes_data/cleaned"
    parquet_files = list(Path(CLEANED_DATA_DIR).rglob("*.parquet"))
    
    if len(parquet_files) == 0:
        logger.error("No parquet files found.")
        return
    
    # Count variable occurrences and distribution
    normalized_counts = {}
    
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        cycle = parquet_file.parent.name
        
        for col in df.columns:
            if col not in normalized_counts:
                normalized_counts[col] = {"cycles": set(), "count": 0}
            
            normalized_counts[col]["cycles"].add(cycle)
            normalized_counts[col]["count"] += df.shape[0]
    
    # Log variables found and their distributions
    logger.info("\nVariables found and their distribution:\n")
    
    for var in sorted(normalized_counts.keys()):
        info = normalized_counts[var]
        cycles_str = ", ".join(sorted(info["cycles"]))
        logger.info(f"  {var:20s} | {info['count']:10,d} rows | Cycles: {cycles_str}")
    
    # Log normalization mappings applied
    logger.info("\n" + "-"*80)
    logger.info("Normalization mappings applied:\n")
    for old_var, normalized_var in sorted(VARIABLE_NORMALIZATION_MAP.items()):
        logger.info(f"  {old_var:15s} → {normalized_var}")
    
    logger.info("="*80)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
