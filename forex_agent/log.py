from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGER_NAME = "forex_agent"
_CONFIGURED = False


def get_logger(name: str = "forex_agent") -> logging.Logger:
    """Return a configured child logger for the given module name.

    The root ``forex_agent`` logger is configured once with a consistent
    stderr handler. No secrets are ever logged.
    """
    global _CONFIGURED
    logger = logging.getLogger(f"{_LOGGER_NAME}.{name}" if name != _LOGGER_NAME else _LOGGER_NAME)
    if not _CONFIGURED:
        _configure_root()
    return logger


def _configure_root(level: Optional[int] = None) -> None:
    global _CONFIGURED
    root = logging.getLogger(_LOGGER_NAME)
    if root.handlers:
        _CONFIGURED = True
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)
    root.setLevel(level or logging.INFO)
    _CONFIGURED = True


def set_level(level: int) -> None:
    """Adjust the logging level at runtime (e.g. from CLI verbosity flags)."""
    _configure_root()
    logging.getLogger(_LOGGER_NAME).setLevel(level)
