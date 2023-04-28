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
        raise RuntimeError("Targets and replacements are different lengths!")

    if not all([t in target_list for t in targets]):
        raise ValueError(f"Targets {targets} not in list: {target_list}")

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

    Returns
    -------
    tuple
        A tuple contain the corresponding array for each input array. The order
        is preserved.
    """
    return tuple(a[indices] for a in args)


def isfinite_struct(x, names=None):
    """Check for +/- infinity and NaNs in a structured array.

    Returns a boolean per entry not per field (name).

    Parameters
    ----------
    x : np.ndarray
        Structured array
    names : list[str]
        Names of the fields to include. If not specified, the names from the
        dtype of the structured array are used.

    Returns
    -------
    np.ndarray
        Array of booleans indicating if each entry in the array is finite
        (True) or not (False).
    """
    if names is None:
        names = x.dtype.names
    return np.all([np.isfinite(x[n]) for n in names], axis=0)


def array_split_chunksize(x, chunksize):
    """Split an array into multiple sub-arrays of a specified chunksize.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    chunksize : int
        Chunksize into which to split the array.

    Returns
    -------
    list
        List of numpy arrays each with a maximum length given by the chunksize.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be greater than 1")
    return np.array_split(x, range(chunksize, len(x), chunksize))


def get_inverse_indices(n, indices):
    """Return the indices that are not in input array given a size n"""
    if indices.max() >= n:
        raise ValueError("Indices contain values that are out of range for n")
    inv = np.arange(n, dtype=int)
    return inv[~np.in1d(inv, indices)]
