"""Loguru configuration for batch correction RNA embeddings project."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

from loguru import logger


def set_logger(
    *,
    level: str = "INFO",
    add_file: bool = False,
    log_file: Union[str, Path] = "logs/run_{time:YYYY-MM-DD_HH-mm-ss}.log",
    file_level: str = "DEBUG",
    rotation: str = "15 MB",
    debug_backtrace: bool = False,
    diagnose: bool = False,
    set_environment: bool = True,
) -> None:
    """
    Configure loguru sinks for interactive / notebook work.

    Parameters
    ----------
    level : str
        Verbosity for the pretty stderr sink.
    add_file : bool
        Whether to add a rotating file sink in addition to stderr.
    log_file : str or Path
        File-path template for the rotating log file.
    file_level : str
        Verbosity for the file sink.
    rotation : str
        Rotation rule passed to ``logger.add``.
    debug_backtrace : bool
        Whether to include ``backtrace=True`` on sinks.
    diagnose : bool
        Whether to include ``diagnose=True`` on sinks.
    set_environment : bool
        Whether to set environment variables required by the pipeline
        (SCIPY_ARRAY_API, PYTORCH_CUDA_ALLOC_CONF).

    Notes
    -----
    Call this **exactly once** per kernel / process before importing
    modules that emit log messages.
    """
    if set_environment:
        os.environ.setdefault("SCIPY_ARRAY_API", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=level,
        enqueue=False,
        backtrace=debug_backtrace,
        diagnose=diagnose,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    if add_file:
        path = Path(str(log_file))
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level=file_level,
            rotation=rotation,
            backtrace=debug_backtrace,
            diagnose=diagnose,
            enqueue=True,
        )
