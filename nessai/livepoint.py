# -*- coding: utf-8 -*-
"""
Functions related to creating live points and converting to other common
data-types.
"""
import numpy as np
from numpy.lib import recfunctions as rfn


LOGL_DTYPE = 'f8'
IT_DTYPE = 'i4'
DEFAULT_FLOAT_DTYPE = 'f8'
CORE_PARAMETERS = ['logP', 'logL', 'it']
DEFAULT_VALUES_CORE = [0.0, 0.0, 0]
EXTRA_PARAMETERS = []
DEFAULT_VALUES_EXTRA = []
NON_SAMPLING_PARAMETERS = CORE_PARAMETERS + EXTRA_PARAMETERS
DEFAULT_VALUES = DEFAULT_VALUES_CORE + DEFAULT_VALUES_EXTRA


def add_extra_parameters_to_live_points(parameters, default_values=None):
    """Add extra parameters to the live points dtype.

    Extra parameters will be included in the live points dtype that is used
    for constructing/converting to/from live points.

    Parameters
    ----------
    parameters: list
        List of parameters to add.
    default_values: list
        List of default values for each parameters. If not specfied, default
        values will be set to zero.
    """
    if default_values is None:
        default_values = len(parameters) * [0.0]
    for p, dv in zip(parameters, default_values):
        if p not in EXTRA_PARAMETERS:
            EXTRA_PARAMETERS.append(p)
            DEFAULT_VALUES_EXTRA.append(dv)


def get_dtype(names, array_dtype=DEFAULT_FLOAT_DTYPE):
    """
    Get a list of tuples containing the dtypes for the structed array

    Parameters
    ----------
    names : list of str
        Names of parameters
    array_dtype : optional
        dtype to use

    Returns
    -------
    list of tuple
        Dtypes as tuples with (field, dtype)
    """
    return (
        [(n, array_dtype) for n in names]
        + [('logP', array_dtype), ('logL', LOGL_DTYPE), ('it', IT_DTYPE)]
        + [(ep, array_dtype) for ep in EXTRA_PARAMETERS]
    )


def live_points_to_array(live_points, names=None):
    """
    Converts live points to unstructered arrays for training.

    Parameters
    ----------
    live_points : structured_array
        Structured array of live points
    names : list of str or None
        If None all fields in the structed array are added to the dictionary
        else only those included in the list are added.

    Returns
    -------
    np.ndarray
        Unstructured numpy array
    """
    if names is None:
        names = list(live_points.dtype.names)
    return rfn.structured_to_unstructured(live_points[names])


def parameters_to_live_point(parameters, names):
    """
    Take a list or array of parameters for a single live point
    and converts them to a live point.

    Returns an empty array with the correct fields if len(parameters) is zero

    Parameters
    ----------
    parameters : tuple
        Float point values for each parameter
    names : tuple
        Names for each parameter as strings

    Returns
    -------
    structed_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if not len(parameters):
        return np.empty(0, dtype=get_dtype(names, DEFAULT_FLOAT_DTYPE))
    else:
        return np.array((*parameters, *DEFAULT_VALUES),
                        dtype=get_dtype(names, DEFAULT_FLOAT_DTYPE))


def numpy_array_to_live_points(array, names):
    """
    Convert a numpy array to a numpy structure array with the correct fields

    Parameters
    ----------
    array : np.ndarray
        Instance of np.ndarray to converto to a structed array
    names : tuple
        Names for each parameter as strings

    Returns
    -------
    structed_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if array.size == 0:
        return np.empty(0, dtype=get_dtype(names))
    if array.ndim == 1:
        array = array[np.newaxis, :]
    struct_array = np.zeros((array.shape[0]), dtype=get_dtype(names))
    for i, n in enumerate(names):
        struct_array[n] = array[..., i]
    return struct_array


def dict_to_live_points(d):
    """
    Convert a dictionary with parameters names as keys to live points.

    Parameters
    ----------
    d : dict
        Dictionary with parmeters names as keys and values that correspond
        to one or more parameters

    Returns
    -------
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if isinstance(list(d.values())[0], int):
        N = 1
    else:
        N = len(list(d.values())[0])
    if N == 1:
        return np.array((*list(d.values()), *DEFAULT_VALUES),
                        dtype=get_dtype(d.keys(), DEFAULT_FLOAT_DTYPE))
    else:
        array = np.zeros(N, dtype=get_dtype(list(d.keys())))
        for k, v in d.items():
            array[k] = v
        return array


def live_points_to_dict(live_points, names=None):
    """
    Convert a structured array of live points to a dictionary with
    a key per field.

    Parameters
    ----------
    live_points : structured_array
        Array of live points
    names : list of str or None
        If None all fields in the structed array are added to the dictionary
        else only those included in the list are added.

    Returns
    -------
    dict
        Dictionary of live points
    """
    if names is None:
        names = live_points.dtype.names
    return {f: live_points[f] for f in names}
