import logging
import os
import traceback
from typing import Literal, Optional

from rich.console import Console
from rich.logging import RichHandler

_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_DEFAULT_LEVEL = "INFO"


class PlainTracebackHandler(RichHandler):
    def formatException(self, exc_info) -> str:
        # Use standard traceback.format_exception() instead of rich.traceback
        return "".join(traceback.format_exception(*exc_info))


def _get_level() -> str:
    level = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid log level: {level}")

    return level


def configure_logger(
    name: str, *, level: Optional[_LEVELS | str] = None
) -> logging.Logger:
    logger = logging.getLogger(name)

    if level is None:
        level = _get_level()
    logger.setLevel(level)

    # Do not add RichHandler if it already exists
    if any(isinstance(handler, RichHandler) for handler in logger.handlers):
        return logger

    console = Console(stderr=True, highlight=True)
    handler = PlainTracebackHandler(
        level=level,
        console=console,
        markup=True,
        show_path=False,
        omit_repeated_times=False,
    )
    formatter = logging.Formatter(
        fmt="%(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
