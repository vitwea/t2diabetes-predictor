"""
Orchestrator Script
=============================================

This script coordinates the complete NHANES data processing pipeline:
1. Download raw XPT files from CDC
2. Clean and normalize the data to Parquet format
3. Merge all cycles into a consolidated dataset

"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Import main functions from each module
from data.download_data import main as download_main
from data.clean_xpt import main as clean_main
from data.merge_data import merge_all_cycles

# Assuming logger utility exists
try:
    from src.utils.logger import get_logger
    logger = get_logger("run_data")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("run_data")


def run_pipeline():
    """
    Execute the complete NHANES data processing pipeline.
    
    Stages:
    -------
    Stage 1: Download
        - Fetches raw XPT files from CDC
        - Saves to: data/nhanes_data/raw/
    
    Stage 2: Clean & Normalize
        - Converts XPT files to Parquet format
        - Normalizes column names across cycles
        - Saves to: data/nhanes_data/cleaned/
    
    Stage 3: Merge
        - Consolidates all cycles
        - Creates binary target variable (diabetes_risk)
        - Saves to: data/nhanes_data/cleaned/dataset_cleaned.parquet
    
    Returns:
    --------
    bool: True if all stages complete successfully, False otherwise
    """
    
    logger.info("=" * 80)
    logger.info("NHANES DATA PIPELINE - START")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {os.path.abspath('.')}")
    
    try:
        # =====================================================================
        # STAGE 1: DOWNLOAD DATA
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: DOWNLOADING DATA FROM CDC")
        logger.info("=" * 80)
        logger.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            download_main()
            logger.info("✓ Download stage completed successfully")
        except Exception as e:
            logger.error(f"✗ Download stage failed: {e}", exc_info=True)
            return False
        
        # =====================================================================
        # STAGE 2: CLEAN & NORMALIZE DATA
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: CLEANING AND NORMALIZING DATA")
        logger.info("=" * 80)
        logger.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            clean_main()
            logger.info("✓ Clean & Normalize stage completed successfully")
        except Exception as e:
            logger.error(f"✗ Clean & Normalize stage failed: {e}", exc_info=True)
            return False
        
        # =====================================================================
        # STAGE 3: MERGE ALL CYCLES
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: MERGING ALL CYCLES")
        logger.info("=" * 80)
        logger.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            df = merge_all_cycles()
            if df is None:
                logger.error("✗ Merge stage returned None")
                return False
            logger.info("✓ Merge stage completed successfully")
        except Exception as e:
            logger.error(f"✗ Merge stage failed: {e}", exc_info=True)
            return False
        
        # =====================================================================
        # PIPELINE COMPLETE
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("NHANES DATA PIPELINE - COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Output file: data/nhanes_data/cleaned/dataset_cleaned.parquet")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Unexpected error in pipeline: {e}", exc_info=True)
        return False


def main():
    """
    Entry point for the pipeline script.
    
    Executes the pipeline and returns appropriate exit code.
    """
    success = run_pipeline()
    
    if success:
        logger.info("\n✓ All stages completed. Pipeline successful!")
        sys.exit(0)
    else:
        logger.error("\n✗ Pipeline failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
