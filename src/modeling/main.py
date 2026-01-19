"""
Main entry point
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.modeling.pipeline import prepare_data_for_modeling

logger = get_logger(__name__)


def main():
    """Execute data preparation"""
    logger.info("Starting data preparation...")
    
    data = prepare_data_for_modeling(
        data_path="C:/Users/pablo/Desktop/t2diabetes-predictor/data/final/nhanes_diabetes.parquet",
        resampling_strategy='smote',
        random_state=42
    )
    
    logger.info(f"✓ Success!")
    logger.info(f"  - X_train: {data['X_train'].shape}")
    logger.info(f"  - X_test: {data['X_test'].shape}")
    
    return data


if __name__ == "__main__":
    data = main()