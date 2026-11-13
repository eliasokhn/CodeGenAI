"""Structured logging helpers shared across the project."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging once with a consistent format.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ...).
        log_file: Optional path; if given, logs are also appended to this file.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level.upper())
        return
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=fmt, handlers=handlers, force=True)
    # Quieten noisy third-party loggers.
    for noisy in ("urllib3", "httpx", "httpx2", "httpcore", "filelock", "fsspec", "docker",
                  "huggingface_hub", "datasets", "hfxet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configures root logging lazily with defaults)."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
