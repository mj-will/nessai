# -*- coding: utf-8 -*-
"""
Test utilities related to logging.
"""

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from nessai.utils.logging import configure_logger


def teardown_function():
    """Reset the logger after each test"""
    logger = logging.getLogger("nessai")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    try:
        os.remove("test.log")
    except OSError:
        pass


@pytest.fixture(params=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
def log_level(request):
    """Different log levels to test."""
    return request.param


def test_configure_logger_no_label():
    """Test behaviour when label is None.

    This should NOT produce a log file.
    """
    logger = configure_logger(label=None)
    assert not any(
        [isinstance(h, logging.FileHandler) for h in logger.handlers]
    )


@pytest.mark.parametrize("output", ["logger_dir", None])
def test_configure_logger_with_label(tmp_path, output):
    """Test behaviour when label is not None.

    This should produce a log file.
    """
    if output:
        output = tmp_path / output
        output.mkdir()
    logger = configure_logger(label="test", output=output)
    if output is None:
        output = os.getcwd()
    assert os.path.exists(os.path.join(output, "test.log"))
    assert any([isinstance(h, logging.FileHandler) for h in logger.handlers])


def test_configure_logger_with_mkdir(tmp_path):
    """Assert the output directory is created if missing"""
    output = tmp_path / "logger_dir"
    configure_logger(label="test", output=output)
    assert os.path.exists(os.path.join(output, "test.log"))


@pytest.mark.parametrize(
    "log_level, value",
    [("ERROR", 40), ("WARNING", 30), ("INFO", 20), ("DEBUG", 10), (15, 15)],
)
def test_configure_logger_levels(log_level, value):
    """Verify logging levels are correctly set."""
    logger = configure_logger(log_level=log_level, label=None)
    assert all([h.level == value for h in logger.handlers])


def test_configure_logger_unknown_level():
    """Verify an error is raised for an unknown level"""
    with pytest.raises(ValueError) as excinfo:
        configure_logger(log_level="test", label=None)
    assert "log_level test not understood" in str(excinfo.value)


def test_filehandler_kwargs(tmp_path, log_level):
    """Assert filehandler kwargs are passed to the handler."""
    output = tmp_path / "logger_dir"
    handler = MagicMock(spec=logging.FileHandler)
    handler.level = 10

    class MockedFileHandler(MagicMock, logging.FileHandler):
        def __new__(cls, *args, **kwargs):
            handler(*args, **kwargs)
            return handler

    with patch("logging.FileHandler", new=MockedFileHandler):
        configure_logger(
            output=output,
            filehandler_kwargs={"mode": "w"},
            log_level=log_level,
        )
    handler.assert_called_once_with(
        os.path.join(output, "nessai.log"),
        mode="w",
    )


def test_filehandler_no_kwargs(tmp_path, log_level):
    """Assert case of no kwargs for the file handler works as intended."""
    output = tmp_path / "logger_dir"
    handler = MagicMock(spec=logging.FileHandler)
    handler.level = 10

    class MockedFileHandler(logging.FileHandler):
        def __new__(cls, *args, **kwargs):
            handler(*args, **kwargs)
            return handler

    with patch("logging.FileHandler", new=MockedFileHandler):
        configure_logger(
            output=output, filehandler_kwargs=None, log_level=log_level
        )
    handler.assert_called_once_with(
        os.path.join(output, "nessai.log"),
    )


@pytest.mark.parametrize(
    "stream, expected",
    (
        [None, None],
        ("stderr", sys.stderr),
        ("stdout", sys.stdout),
        (sys.stderr, sys.stderr),
    ),
)
def test_stream_handler_setting(tmp_path, stream, expected, log_level):
    output = tmp_path / "logger_dir"
    handler = MagicMock(spec=logging.StreamHandler)
    handler.level = 10

    class MockedStreamHandler(logging.StreamHandler):
        def __new__(cls, *args, **kwargs):
            handler(*args, **kwargs)
            return handler

    with patch("logging.StreamHandler", new=MockedStreamHandler):
        configure_logger(
            output=output, stream=stream, label=None, log_level=log_level
        )
    handler.assert_called_with(expected)


def test_stream_handler_error(tmp_path):
    """Assert an error is raised if an invalid string is passes"""
    output = tmp_path / "logger_dir"
    with pytest.raises(ValueError, match=r"Unknown stream: .*"):
        configure_logger(output=output, stream="not_a_stream")


@pytest.mark.parametrize("include_logger_name", [True, False])
def test_configure_logger_include_logger_name(tmp_path, include_logger_name):
    output = tmp_path / "logger_dir"
    configure_logger(output=output, include_logger_name=include_logger_name)

    logger = logging.getLogger("nessai")
    if include_logger_name:
        assert all(
            [
                h.formatter._fmt
                == "%(asctime)s %(name)s %(levelname)-8s: %(message)s"
                for h in logger.handlers
                if h.formatter is not None
            ]
        )
    else:
        assert all(
            [
                h.formatter._fmt
                == "%(asctime)s nessai %(levelname)-8s: %(message)s"
                for h in logger.handlers
                if h.formatter is not None
            ]
        )
