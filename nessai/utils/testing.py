# -*- coding: utf-8 -*-
"""
Utilities for the test suite.
"""
import numpy as np


def assert_structured_arrays_equal(x, y, atol=0.0, rtol=0.0):
    """Assert structured arrays are equal.

    Supports NaNs by checking each field individually.

    Parameters
    ----------
    x : np.ndarray
        Array to check.
    y : np.ndarray
        Array to compare to.
    atol : float
        The absolute tolerance parameter. See the numpy documentation for all
        :code:`numpy.allclose`.
    rtol : float
        The relative tolerance parameter. See the numpy documentation for all
        :code:`numpy.allclose`.

    Raises
    ------
    AssertionError
        If dtype or values in each field are not equal.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    if x.dtype != y.dtype:
        raise AssertionError(
            f"""Structured array dtypes are different:

            Expected: {x.dtype}
            Actual: {y.dtype}
            """
        )

    valid = {f: False for f in x.dtype.names}
    for field in valid.keys():
        valid[field] = np.allclose(
            x[field], y[field], equal_nan=True, atol=atol, rtol=rtol
        )

    if not all(valid.values()):
        mismatched = [k for k, v in valid.items() if v is False]
        msg = f"""
        Arrays are not equal.

        Mismatched fields: {mismatched}
        """
        for f in mismatched:
            msg += f"""
            Field: {f}
            Expected: {y[f]}
            Actual: {x[f]}
            """
        raise AssertionError(msg)
