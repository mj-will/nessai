# -*- coding: utf-8 -*-
"""Test the version number."""
from importlib import reload
from unittest.mock import patch

try:
    from importlib.metadata import PackageNotFoundError

    LIB = "importlib.metadata"
except ImportError:
    from importlib_metadata import PackageNotFoundError

    LIB = "importlib_metadata"

import nessai
import pytest


@pytest.fixture(autouse=True)
def reload_nessai():
    """Make sure nessai is reloaded after these tests"""
    original_version = nessai.__version__
    yield
    reload(nessai)
    assert nessai.__version__ == original_version


def test_nessai_version():
    """Assert the correct version is set"""
    with patch(f"{LIB}.version", return_value="1.2.3"):
        reload(nessai)
    assert nessai.__version__ == "1.2.3"


def test_nessai_version_package_not_found():
    """Assert the version is unknown if the package is not installed."""
    with patch(f"{LIB}.version", side_effect=PackageNotFoundError):
        reload(nessai)
    assert nessai.__version__ == "unknown"
