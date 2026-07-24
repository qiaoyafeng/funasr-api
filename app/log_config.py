"""Loguru logging configuration with console and date-named file output."""

import os
import sys
import logging

from loguru import logger

# Log directory (configurable via environment variable)
LOG_DIR = os.getenv("FUNASR_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Remove default loguru handler
logger.remove()

# Console handler (colorized)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
    colorize=True,
)

# File handler (date-named, daily rotation, 30-day retention)
logger.add(
    os.path.join(LOG_DIR, "funasr-api_{time:YYYY-MM-DD}.log"),
    level="INFO",
    rotation="00:00",
    retention="30 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)


class InterceptHandler(logging.Handler):
    """Forward standard logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Uvicorn log config that routes all loggers through InterceptHandler -> loguru
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "default": {
            "class": "app.log_config.InterceptHandler",
        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}
