# -*- coding: utf-8 -*-
"""
Test utilities related to logging.
"""
import logging
import os
import pytest

from nessai.utils.logging import setup_logger


def teardown_function():
    """Reset the logger after each test"""
    logger = logging.getLogger('nessai')
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    try:
        os.remove('test.log')
    except OSError:
        pass


def test_setup_logger_no_label():
    """Test behaviour when label is None.

    This should NOT produce a log file.
    """
    logger = setup_logger(label=None)
    assert not any([type(h) == logging.FileHandler for h in logger.handlers])


@pytest.mark.parametrize('output', ['logger_dir', None])
def test_setup_logger_with_label(tmpdir, output):
    """Test behaviour when label is not None.

    This should produce a log file.
    """
    if output:
        output = str(tmpdir.mkdir(output))
    logger = setup_logger(label='test', output=output)
    if output is None:
        output = '.'
    assert os.path.exists(f'{output}/test.log')
    assert any([type(h) == logging.FileHandler for h in logger.handlers])


def test_setup_logger_with_mkdir(tmpdir):
    """Assert the output directory is created if missing"""
    output = str(tmpdir) + '/logger_dir/'
    setup_logger(label='test', output=output)
    assert os.path.exists(f'{output}/test.log')


@pytest.mark.parametrize(
    'log_level, value',
    [('ERROR', 40), ('WARNING', 30), ('INFO', 20), ('DEBUG', 10), (15, 15)]
)
def test_setup_logger_levels(log_level, value):
    """Verify logging levels are correctly set."""
    logger = setup_logger(log_level=log_level, label=None)
    assert all([h.level == value for h in logger.handlers])


def test_setup_logger_unknown_level():
    """Verify an error is raised for an unknown level"""
    with pytest.raises(ValueError) as excinfo:
        setup_logger(log_level='test', label=None)
    assert 'log_level test not understood' in str(excinfo.value)
