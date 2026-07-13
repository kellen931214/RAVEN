import logging
import os
from datetime import datetime

def get_logger(log_dir, log_name="reprompting", level=logging.INFO):
    """
    Build and return a logger, and output to file and console。

    Args:
        log_dir (str): log folder path
        log_name (str): logger name & log file name prefix
        level (int): logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # avoid add handler repeatedly when calling get_logger multiple times
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # 格式
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
