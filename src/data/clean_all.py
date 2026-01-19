import os
import glob
import logging
from clean_xpt import clean_xpt

# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

RAW_DIR = "./data/nhanes_data"
CLEAN_DIR = "./data/cleaned"

def main():
    """
    Iterate through all downloaded NHANES cycles, clean each XPT file,
    and export the cleaned version as Parquet.
    """

    logging.info("Starting NHANES cleaning pipeline")

    cycles = sorted(os.listdir(RAW_DIR))

    for cycle in cycles:
        cycle_path = os.path.join(RAW_DIR, cycle)

        if not os.path.isdir(cycle_path):
            continue

        logging.info(f"Processing cycle: {cycle}")

        # Find all XPT files inside the cycle folder
        xpt_files = glob.glob(os.path.join(cycle_path, "**/*.xpt"), recursive=True)

        if not xpt_files:
            logging.warning(f"No XPT files found in {cycle_path}")
            continue

        for xpt in xpt_files:
            filename = os.path.splitext(os.path.basename(xpt))[0]
            output_folder = os.path.join(CLEAN_DIR, cycle)
            output_path = os.path.join(output_folder, f"{filename}.parquet")

            clean_xpt(xpt, output_path)

    logging.info("Cleaning pipeline completed")

if __name__ == "__main__":
    main()