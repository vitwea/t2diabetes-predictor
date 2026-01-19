import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Creates and configures a logger with both file and console handlers.
    Ensures reproducible, clean, and professional logging across the project.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the module).
    log_dir : str
        Directory where log files will be stored.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Build log file path with timestamp for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filepath = os.path.join(log_dir, f"{name.replace('.', '_')}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger