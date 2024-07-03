import logging
import sys
from copy import copy, deepcopy
from logging.config import dictConfig
from typing import Any, Dict, Literal, Optional

import click

logging_config = dict(
    version=1,
    # when disabling is true and other libraries are
    # important before dictConfig is called,
    # then their logging messages will be lost
    # instead of propagated to the root logger.
    disable_existing_loggers=False,
    formatters={
        "f": {
            "()": "anomalib_wrapper.utils.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(name)-12s  %(message)s",
            "use_colors": True,
        }
    },
    handlers={
        "default_stdout": {
            "class": "logging.StreamHandler",
            "formatter": "f",
            "level": logging.INFO,
        }
    },
    root={
        "handlers": ["default_stdout"],
        "level": logging.INFO,
    },
)


TRACE_LOG_LEVEL = 5


class DefaultFormatter(logging.Formatter):
    """
    Source: https://github.com/encode/uvicorn/blob/eaf8b4d7a52374712d71afd5049a13f70267130a/uvicorn/logging.py  # noqa: E501
    A custom log formatter class that:
    * Outputs the LOG_LEVEL with an appropriate color.
    * If a log call includes an `extras={"color_message": ...}` it will be used
      for formatting the output, instead of the plain text message.
    """

    level_name_colors = {
        TRACE_LOG_LEVEL: lambda level_name: click.style(str(level_name), fg="blue"),
        logging.DEBUG: lambda level_name: click.style(str(level_name), fg="cyan"),
        logging.INFO: lambda level_name: click.style(str(level_name), fg="green"),
        logging.WARNING: lambda level_name: click.style(str(level_name), fg="yellow"),
        logging.ERROR: lambda level_name: click.style(str(level_name), fg="red"),
        logging.CRITICAL: lambda level_name: click.style(str(level_name), fg="bright_red"),
    }

    def __init__(
            self,
            fmt: Optional[str] = None,
            datefmt: Optional[str] = None,
            style: Literal["%", "{", "$"] = "%",
            use_colors: Optional[bool] = None,
    ):
        if use_colors in (True, False):
            self.use_colors = use_colors
        else:
            self.use_colors = sys.stdout.isatty()
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def color_level_name(self, level_name: str, level_no: int) -> str:
        def default(level_name: str) -> str:
            return str(level_name)  # pragma: no cover

        func = self.level_name_colors.get(level_no, default)
        return func(level_name)

    def should_use_colors(self) -> bool:
        return True  # pragma: no cover

    def formatMessage(self, record: logging.LogRecord) -> str:
        recordcopy = copy(record)
        levelname = recordcopy.levelname
        seperator = " " * (8 - len(recordcopy.levelname))
        if self.use_colors:
            levelname = self.color_level_name(levelname, recordcopy.levelno)
            if "color_message" in recordcopy.__dict__:
                recordcopy.msg = recordcopy.__dict__["color_message"]
                recordcopy.__dict__["message"] = recordcopy.getMessage()
        recordcopy.__dict__["levelprefix"] = levelname + ":" + seperator
        return super().formatMessage(recordcopy)


def generate_base_dict_config():
    return deepcopy(logging_config)


def overwrite_log_level(config: Dict[str, Any], log_level: Optional[Literal[20]] = None):
    if log_level is not None:
        config["root"]["level"] = log_level
        if "default_stdout" in config["handlers"]:
            config["handlers"]["default_stdout"]["level"] = log_level


def setup_logging(log_level: Optional[Literal[20]] = None):
    """
    Sets up logging for scripts and applications.
    Should never be called by library functions of CVC.
    Only in entrypoints or user scripts.

    TODO: allow specifying dictionary or config file for configuration.

    :param log_level: optionally overwrite the log level of the root
        logger and the default stdout handler.
        Does not affect log level of other handlers.
    """
    config = generate_base_dict_config()
    overwrite_log_level(config, log_level=log_level)
    dictConfig(config)


def add_log_file_handler(log_file: str):
    # Create a FileHandler for the logfile
    file_handler = logging.FileHandler(log_file)

    # Create a Formatter to specify the log message format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Get the root logger
    root_logger = logging.getLogger()

    # Add the FileHandler to the root logger
    root_logger.addHandler(file_handler)
