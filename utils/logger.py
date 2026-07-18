"""Structured logger used across all modules."""

import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from config import LOG_DIR

console = Console()

def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with RichHandler and a rotating file sink."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler (rich)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)

    # File handler
    log_file = LOG_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    return logger
