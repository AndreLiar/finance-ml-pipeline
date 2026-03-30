"""
logger.py — Centralised rotating logger for the finance ML pipeline.

Usage in any script:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Loaded %d transactions", len(df))
    log.warning("Validation failed: %s", msg)
    log.error("Stage crashed: %s", err)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Logs land in <project_root>/logs/pipeline.log
_LOGS_DIR  = Path(__file__).parent / "logs"
_LOG_FILE  = _LOGS_DIR / "pipeline.log"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 3               # keep pipeline.log, .1, .2, .3

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Rotating file handler — DEBUG and above
    fh = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    root.addHandler(fh)
    root.addHandler(ch)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring handlers on first call."""
    _configure()
    return logging.getLogger(name)
