"""Lightweight logging setup shared by pipeline scripts."""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int | None = None) -> logging.Logger:
    """Configure root logging for scripts.

    Reads ``LOG_LEVEL`` from the environment if ``level`` is not provided.
    Safe to call multiple times — subsequent calls update the level only.
    """
    resolved = level if level is not None else os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = resolved.upper()

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=resolved,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            stream=sys.stdout,
        )
    else:
        root.setLevel(resolved)

    return logging.getLogger("aqml")
