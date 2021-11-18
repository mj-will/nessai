# -*- coding: utf-8 -*-
"""
Utilities for manipulating python structures such as lists and dictionaries.
"""
import numpy as np


def replace_in_list(target_list, targets, replacements):
    """
    Replace (in place) an entry in a list with a given element.

    Parameters
    ----------
    target_list : list
        List to update
    targets : list
        List of items to update
    replacements : list
        List of replacement items
    """
    if not isinstance(targets, list):
        targets = [targets]
    if not isinstance(replacements, list):
        replacements = [replacements]

    if not len(targets) == len(replacements):
        raise RuntimeError('Targets and replacements are different lengths!')

    if not all([t in target_list for t in targets]):
        raise ValueError(f'Targets {targets} not in list: {target_list}')

    for t, r in zip(targets, replacements):
        i = target_list.index(t)
        target_list[i] = r


def get_subset_arrays(indices, *args):
    """Return a subset of a set of arrays.

    Assumes all arrays are the same length.

    Parameters
    ----------
    indices : array
        Array of indices or boolean array of same length as input arrays
    args : arrays
        Set of arrays to index.
    """
    return (a[indices] for a in args)


def isfinite_struct(x, names=None):
    """Check for +/- infinity and NaNs in a structued array.

    Parameters
    ----------
    x : np.ndarray
        Structured array
    names : list[str]
        Names of the fields to include. If not specified all fields are
        checked.

    Returns
    -------
    np.ndarray
        Array of booleans indicating if each entry in the array is finite
        (True) or not (False).
    """
    if names is None:
        names = x.dtype.names
    return np.all([np.isfinite(x[n]) for n in names], axis=0)
