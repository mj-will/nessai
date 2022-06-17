# -*- coding: utf-8 -*-
"""
Utilities for the test suite.
"""
import numpy as np


def assert_structured_arrays_equal(x, y):
    """Assert structured arrays are equal.

    Supports NaNs by checking each field individually.

    Parameters
    ----------
    x : np.ndarray
        Array to check.
    y : np.ndarray
        Array to compare to.

    Raises
    ------
    AssertionError
        If dtype or values in each field are not equal.
    """
    if x.dtype != y.dtype:
        raise AssertionError(
            f"""Structured array dtypes are different:

            Expected: {x.dtype}
            Actual: {y.dtype}
            """
        )

    valid = {f: False for f in x.dtype.names}
    for field in valid.keys():
        valid[field] = np.array_equal(x[field], y[field], equal_nan=True)

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
