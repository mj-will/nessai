# -*- coding: utf-8 -*-
"""
Functions related to creating live points and converting to other common
data-types.
"""
import logging

import numpy as np
from numpy.lib import recfunctions as rfn

from . import config

logger = logging.getLogger(__name__)


def add_extra_parameters_to_live_points(parameters, default_values=None,):
    """Add extra parameters to the live points dtype.

    Extra parameters will be included in the live points dtype that is used
    for constructing/converting to/from live points.

    Parameters
    ----------
    parameters: list
        List of parameters to add.
    default_values: Optional[list]
        List of default values for each parameters. If not specified, default
        values will be set to based on :code: `DEFAULT_FLOAT_VALUE` in
        :code:`nessai.config`.
    """
    if default_values is None:
        default_values = len(parameters) * [config.DEFAULT_FLOAT_VALUE]
    for p, dv in zip(parameters, default_values):
        if p not in config.EXTRA_PARAMETERS:
            config.EXTRA_PARAMETERS.append(p)
            config.EXTRA_PARAMETERS_DEFAULTS.append(dv)
            config.EXTRA_PARAMETERS_DTYPE.append(config.DEFAULT_FLOAT_DTYPE)
    config.NON_SAMPLING_PARAMETERS = \
        config.CORE_PARAMETERS + config.EXTRA_PARAMETERS
    config.NON_SAMPLING_DEFAULTS = \
        config.CORE_PARAMETERS_DEFAULTS + config.EXTRA_PARAMETERS_DEFAULTS
    config.NON_SAMPLING_DEFAULT_DTYPE = \
        config.CORE_PARAMETERS_DTYPE + config.EXTRA_PARAMETERS_DTYPE
    logger.debug(
        f'Updated non-sampling parameters: {config.NON_SAMPLING_PARAMETERS}'
    )
    logger.debug(
        'Updated defaults for non-sampling parameters: '
        f'{config.NON_SAMPLING_DEFAULTS}'
    )


def reset_extra_live_points_parameters():
    """Reset the extra live points parameters."""
    logger.debug('Resetting extra parameters')
    config.EXTRA_PARAMETERS = []
    config.EXTRA_PARAMETERS_DEFAULTS = []
    config.EXTRA_PARAMETERS_DTYPE = []
    config.NON_SAMPLING_PARAMETERS = \
        config.CORE_PARAMETERS + config.EXTRA_PARAMETERS
    config.NON_SAMPLING_DEFAULTS = \
        config.CORE_PARAMETERS_DEFAULTS + config.EXTRA_PARAMETERS_DEFAULTS
    config.NON_SAMPLING_DEFAULT_DTYPE = \
        config.CORE_PARAMETERS_DTYPE + config.EXTRA_PARAMETERS_DTYPE


def get_dtype(names, array_dtype=config.DEFAULT_FLOAT_DTYPE):
    """
    Get a list of tuples containing the dtypes for the structured array

    Parameters
    ----------
    names : list of str
        Names of parameters
    array_dtype : optional
        dtype to use

    Returns
    -------
    numpy.dtype
        A instance of :code:`numpy.dtype`.
    """
    return np.dtype(
        [(n, array_dtype) for n in names]
        + list(zip(
            config.NON_SAMPLING_PARAMETERS,
            config.NON_SAMPLING_DEFAULT_DTYPE,
        ))
    )


def empty_structured_array(n, names=None, dtype=None):
    """Get an empty structured array with the extra parameters initialised.

    Parameters
    ----------
    n : int
        Length of the structured array
    dtype : Optional[list]
        Dtype to use. Must contain the non-sampling parameters.
    names : Optional[list]
        Names of fields (excluding non-sampling parameters) to construct the
        dtype. Must be specified if :code:`dtype` is not specified.

    Returns
    -------
    np.ndarray
        Structured array with the all parameters initialised to their
        default values.
    """
    if dtype is None:
        dtype = get_dtype(names)
    else:
        dtype = np.dtype(dtype)
        names = [
            nm for nm in dtype.names
            if nm not in config.NON_SAMPLING_PARAMETERS
        ]
    struct_array = np.empty((n), dtype=dtype)
    if n == 0:
        return struct_array
    struct_array[names] = config.DEFAULT_FLOAT_VALUE
    try:
        for nm, v in zip(
            config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULTS
        ):
            struct_array[nm] = v
    except ValueError:
        raise ValueError(
            "Could not create empty structured array. Maybe the non-sampling "
            "parameters are missing?"
        )
    return struct_array


def live_points_to_array(live_points, names=None):
    """
    Converts live points to unstructured arrays for training.

    Parameters
    ----------
    live_points : structured_array
        Structured array of live points
    names : list of str or None
        If None all fields in the structured array are added to the dictionary
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
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if not len(parameters):
        return empty_structured_array(0, names)
    else:
        return np.array([(*parameters, *config.NON_SAMPLING_DEFAULTS)],
                        dtype=get_dtype(names, config.DEFAULT_FLOAT_DTYPE))


def numpy_array_to_live_points(array, names):
    """
    Convert a numpy array to a numpy structure array with the correct fields

    Parameters
    ----------
    array : np.ndarray
        Instance of np.ndarray to convert to a structured array
    names : tuple
        Names for each parameter as strings

    Returns
    -------
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if array.size == 0:
        return empty_structured_array(0, names=names)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    struct_array = empty_structured_array(len(array), names=names)
    for i, n in enumerate(names):
        struct_array[n] = array[..., i]
    return struct_array


def dict_to_live_points(d):
    """Convert a dictionary with parameters names as keys to live points.

    Assumes all entries have the same length. Also, determines number of points
    from the first entry by checking if the value has `__len__` attribute,
    if not the dictionary is assumed to contain a single point.

    Parameters
    ----------
    d : dict
        Dictionary with parameters names as keys and values that correspond
        to one or more parameters

    Returns
    -------
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    a = list(d.values())
    if hasattr(a[0], '__len__'):
        N = len(a[0])
    else:
        N = 1
    if N == 1:
        return np.array([(*a, *config.NON_SAMPLING_DEFAULTS)],
                        dtype=get_dtype(d.keys(), config.DEFAULT_FLOAT_DTYPE))
    else:
        array = empty_structured_array(N, names=list(d.keys()))
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
        If None all fields in the structured array are added to the dictionary
        else only those included in the list are added.

    Returns
    -------
    dict
        Dictionary of live points
    """
    if names is None:
        names = live_points.dtype.names
    return {f: live_points[f] for f in names}


def dataframe_to_live_points(df):
    """Convert and pandas dataframe to live points.

    Adds the additional parameters logL and logP initialised to zero.

    Based on this answer on Stack Exchange:
    https://stackoverflow.com/a/51280608

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Pandas DataFrame to convert to live points

    Returns
    -------
    structured_array
        Numpy structured array with fields given by column names plus logP and
        logL.
    """
    return np.array(
        [tuple(x) + tuple(config.NON_SAMPLING_DEFAULTS) for x in df.values],
        dtype=get_dtype(list(df.dtypes.index))
    )
