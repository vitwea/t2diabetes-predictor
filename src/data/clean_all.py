import os
import glob

from src.data.clean_xpt import clean_xpt
from src.utils.logger import get_logger

# ---------------------------------------------------------
# Logging configuration 
# ---------------------------------------------------------
logger = get_logger("nhanes_cleaning")

RAW_DIR = "./data/nhanes_data"
CLEAN_DIR = "./data/cleaned"

def main():
    """
    Iterate through all downloaded NHANES cycles, clean each XPT file,
    and export the cleaned version as Parquet.
    """

    logger.info("Starting NHANES cleaning pipeline")

    cycles = sorted(os.listdir(RAW_DIR))

    for cycle in cycles:
        cycle_path = os.path.join(RAW_DIR, cycle)

        if not os.path.isdir(cycle_path):
            continue

        logger.info(f"Processing cycle: {cycle}")

        # Find all XPT files inside the cycle folder
        xpt_files = glob.glob(os.path.join(cycle_path, "**/*.xpt"), recursive=True)

        if not xpt_files:
            logger.warning(f"No XPT files found in {cycle_path}")
            continue

        for xpt in xpt_files:
            filename = os.path.splitext(os.path.basename(xpt))[0]
            output_folder = os.path.join(CLEAN_DIR, cycle)
            output_path = os.path.join(output_folder, f"{filename}.parquet")

            clean_xpt(xpt, output_path)

    logger.info("Cleaning pipeline completed")

if __name__ == "__main__":
    main()