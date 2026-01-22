import pandas as pd
import glob
import os

from src.utils.logger import get_logger 

# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------
logger = get_logger("merge_cycles")

# ---------------------------------------------------------
# Merge all Parquet files for a given cycle
# ---------------------------------------------------------
def merge_cycle(cycle_folder):
    """
    Merge all cleaned Parquet files inside a cycle folder using 'id' as key.
    """

    logger.info(f"Starting merge for cycle folder: {cycle_folder}")

    files = glob.glob(os.path.join(cycle_folder, "*.parquet"))
    logger.info(f"Found {len(files)} cleaned files")

    if not files:
        logger.warning(f"No Parquet files found in {cycle_folder}")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            logger.info(f"Loaded {os.path.basename(f)} with {df.shape[0]} rows")
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    # Merge all dataframes on 'id'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="id", how="outer")

    logger.info(f"Merged cycle dataset shape: {merged.shape}")

    return merged

# ---------------------------------------------------------
# Main pipeline: merge all cycles
# ---------------------------------------------------------
def main():
    base = "./data/cleaned/"
    cycles = sorted(os.listdir(base))

    logger.info("Starting full NHANES merge pipeline")
    all_cycles = []

    for cycle in cycles:
        cycle_path = os.path.join(base, cycle)

        if not os.path.isdir(cycle_path):
            continue

        logger.info(f"Processing cycle: {cycle}")

        merged = merge_cycle(cycle_path)

        if merged is not None:
            merged["cycle"] = cycle
            all_cycles.append(merged)

    if not all_cycles:
        logger.error("No cycles were successfully merged")
        return

    final = pd.concat(all_cycles, ignore_index=True)
    logger.info(f"Final dataset shape: {final.shape}")

    output_path = "./data/final/nhanes_diabetes.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.to_parquet(output_path, index=False)

    logger.info(f"Final dataset saved to: {output_path}")

if __name__ == "__main__":
    main()