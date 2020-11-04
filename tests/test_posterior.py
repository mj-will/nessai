import pytest

from nessai.posterior import (
        logsubexp
        )


def test_logsubexp_negative():
    """
    Test behaviour of logsubexp for x < y
    """
    with pytest.raises(Exception):
        logsubexp(1, 2)
