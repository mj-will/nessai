# -*- coding: utf-8 -*-
"""
Utilities related to logging.
"""

import logging
import os
import sys


def configure_logger(
    output=None,
    label="nessai",
    log_level="INFO",
    filehandler_kwargs=None,
    include_logger_name=False,
    stream=None,
):
    """
    Configure the logger.

    Based on the implementation in Bilby:
    https://github.com/bilby-dev/bilby/blob/main/bilby/core/utils/log.py

    .. versionchanged:: 0.14.0
        Renamed :code:`setup_logger` to :code:`configure_logger`.

    .. versionadded:: 0.14.0
        Added :code:`include_logger_name` argument.

    Parameters
    ----------
    output : str, optional
        Path of to output directory.
    label : str, optional
        Label for this instance of the logger.
    log_level : {'ERROR', 'WARNING', 'INFO', 'DEBUG'}, optional
        Level of logging passed to logger.
    filehandler_kwargs : dict, optional
        Keyword arguments for configuring the FileHandler. See logging
        documentation for details.
    include_logger_name : bool, optional
        If true, include the logger name in the log output. If false, only
        the name will be replaced with 'nessai'.
    stream : str, file-object, optional
        Stream passes to :code:`logging.StreamHandler` to set the stream. See
        logging documentation for more details.

    Returns
    -------
    :obj:`logging.Logger`
        Instance of the Logger class.
    """
    from .. import __version__ as version

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError("log_level {} not understood".format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger("nessai")
    logger.setLevel(level)

    if include_logger_name:
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)-8s: %(message)s",
            datefmt="%m-%d %H:%M",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s nessai %(levelname)-8s: %(message)s",
            datefmt="%m-%d %H:%M",
        )

    if (
        any([isinstance(h, logging.StreamHandler) for h in logger.handlers])
        is False
    ):
        if isinstance(stream, str):
            if stream.lower() == "stderr":
                stream = sys.stderr
            elif stream.lower() == "stdout":
                stream = sys.stdout
            else:
                raise ValueError(
                    f"Unknown stream: {stream}. Choose from: [stderr, stdout]"
                )
        stream_handler = logging.StreamHandler(stream)

        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if (
        any([isinstance(h, logging.FileHandler) for h in logger.handlers])
        is False
    ):
        if label:
            if output:
                if not os.path.exists(output):
                    os.makedirs(output, exist_ok=True)
            else:
                output = os.getcwd()
            log_file = os.path.join(output, f"{label}.log")
            if filehandler_kwargs is None:
                filehandler_kwargs = {}
            file_handler = logging.FileHandler(log_file, **filehandler_kwargs)
            file_handler.setFormatter(formatter)

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info(f"Running Nessai version {version}")

    return logger


def setup_logger(*args, **kwargs):
    """
    Wrapper for configure_logger to maintain backwards compatibility.

    .. deprecated:: 0.14.0
        Use :func:`configure_logger` instead.
    """
    import warnings

    warnings.warn(
        "setup_logger is deprecated, use configure_logger instead",
        FutureWarning,
    )
    return configure_logger(*args, **kwargs)
