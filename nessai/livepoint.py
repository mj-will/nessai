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


def add_extra_parameters_to_live_points(
    parameters,
    default_values=None,
):
    """Add extra parameters to the live points dtype.

    Extra parameters will be included in the live points dtype that is used
    for constructing/converting to/from live points.

    Parameters
    ----------
    parameters: list
        List of parameters to add.
    default_values: Optional[Union[List, Tuple]]
        List of default values for each parameters. If not specified, default
        values will be set to based on :code: `DEFAULT_FLOAT_VALUE` in
        :code:`nessai.config`.
    """
    if default_values is None:
        default_values = len(parameters) * (
            config.livepoints.default_float_value,
        )
    else:
        default_values = tuple(default_values)
    for p, dv in zip(parameters, default_values):
        if p not in config.livepoints.extra_parameters:
            config.livepoints.extra_parameters.append(p)
            config.livepoints.extra_parameters_defaults = (
                config.livepoints.extra_parameters_defaults + (dv,)
            )
            config.livepoints.extra_parameters_dtype.append(
                config.livepoints.default_float_dtype
            )
        else:
            logger.warning(
                f"Extra parameter `{p}` has already been added. Skipping."
                "Call `reset_extra_live_points_parameters` to reset the values"
                " and add this parameter."
            )

    logger.debug(
        "Updated non-sampling parameters: "
        f"{config.livepoints.non_sampling_parameters}"
    )
    logger.debug(
        "Updated defaults for non-sampling parameters: "
        f"{config.livepoints.non_sampling_defaults}"
    )
    config.livepoints.reset_properties()


def reset_extra_live_points_parameters():
    """Reset the extra live points parameters."""
    logger.debug("Resetting extra parameters")
    config.livepoints.reset()


def get_dtype(names, array_dtype=None, non_sampling_parameters=True):
    """
    Get a list of tuples containing the dtypes for the structured array

    Parameters
    ----------
    names : list of str
        Names of parameters
    array_dtype : Optional[str]
        dtype to use
    non_sampling_parameters : bool
        Indicates whether non-sampling parameters should be included.

    Returns
    -------
    numpy.dtype
        A instance of :code:`numpy.dtype`.
    """
    if array_dtype is None:
        array_dtype = config.livepoints.default_float_dtype
    dtype = [(n, array_dtype) for n in names]
    if non_sampling_parameters:
        dtype += list(
            zip(
                config.livepoints.non_sampling_parameters,
                config.livepoints.non_sampling_dtype,
            )
        )
    return np.dtype(dtype)


def empty_structured_array(
    n, names=None, dtype=None, non_sampling_parameters=True
):
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
    non_sampling_parameters : bool
        Indicates whether non-sampling parameters should be included.

    Returns
    -------
    np.ndarray
        Structured array with the all parameters initialised to their
        default values.
    """
    if dtype is None:
        dtype = get_dtype(
            names, non_sampling_parameters=non_sampling_parameters
        )
    else:
        dtype = np.dtype(dtype)
        names = [
            nm
            for nm in dtype.names
            if nm not in config.livepoints.non_sampling_parameters
        ]
    struct_array = np.empty((n), dtype=dtype)
    if n == 0:
        return struct_array
    struct_array[names] = config.livepoints.default_float_value
    if non_sampling_parameters:
        try:
            for nm, v in zip(
                config.livepoints.non_sampling_parameters,
                config.livepoints.non_sampling_defaults,
            ):
                struct_array[nm] = v
        except ValueError:
            raise ValueError(
                "Could not create empty structured array. Maybe the "
                "non-sampling parameters are missing?"
            )
    return struct_array


def live_points_to_array(live_points, names=None, copy=False):
    """
    Converts live points to unstructured arrays for training.

    Parameters
    ----------
    live_points : structured_array
        Structured array of live points
    names : list of str or None
        If None all fields in the structured array are added to the dictionary
        else only those included in the list are added.
    copy : bool
        If true, returns a copy. If false, returns a view. See numpy
        documentation for
        :code:`numpy.lib.recfunctions.structured_to_unstructured` for more
        details.

    Returns
    -------
    np.ndarray
        Unstructured numpy array
    """
    if names is None:
        names = list(live_points.dtype.names)
    return rfn.structured_to_unstructured(live_points[names], copy=copy)


def parameters_to_live_point(parameters, names, non_sampling_parameters=True):
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
    non_sampling_parameters : bool
        Indicates whether non-sampling parameters should be included.

    Returns
    -------
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if not len(parameters):
        return empty_structured_array(
            0, names, non_sampling_parameters=non_sampling_parameters
        )
    else:
        if non_sampling_parameters:
            parameters = [
                (*parameters, *config.livepoints.non_sampling_defaults)
            ]
        else:
            parameters = [tuple(parameters)]
        return np.array(
            parameters,
            dtype=get_dtype(
                names,
                config.livepoints.default_float_dtype,
                non_sampling_parameters=non_sampling_parameters,
            ),
        )


def numpy_array_to_live_points(array, names, non_sampling_parameters=True):
    """
    Convert a numpy array to a numpy structure array with the correct fields

    Parameters
    ----------
    array : np.ndarray
        Instance of np.ndarray to convert to a structured array
    names : tuple
        Names for each parameter as strings
    non_sampling_parameters : bool
        Indicates whether non-sampling parameters should be included.

    Returns
    -------
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    if array.size == 0:
        return empty_structured_array(
            0, names=names, non_sampling_parameters=non_sampling_parameters
        )
    if array.ndim == 1:
        array = array[np.newaxis, :]
    struct_array = empty_structured_array(
        len(array),
        names=names,
        non_sampling_parameters=non_sampling_parameters,
    )
    for i, n in enumerate(names):
        struct_array[n] = array[..., i]
    return struct_array


def dict_to_live_points(d, non_sampling_parameters=True):
    """Convert a dictionary with parameters names as keys to live points.

    Assumes all entries have the same length. Also, determines number of points
    from the first entry by checking if the value has `__len__` attribute,
    if not the dictionary is assumed to contain a single point.

    Parameters
    ----------
    d : dict
        Dictionary with parameters names as keys and values that correspond
        to one or more parameters
    non_sampling_parameters : bool
        Indicates whether non-sampling parameters should be included.

    Returns
    -------
    structured_array
        Numpy structured array with fields given by names plus logP and logL
    """
    a = tuple(d.values())
    if hasattr(a[0], "__len__"):
        N = len(a[0])
    else:
        N = 1
    if N == 1:
        if non_sampling_parameters:
            a = (*a, *config.livepoints.non_sampling_defaults)
        return np.array(
            [a],
            dtype=get_dtype(
                d.keys(),
                config.livepoints.default_float_dtype,
                non_sampling_parameters=non_sampling_parameters,
            ),
        )
    else:
        array = empty_structured_array(
            N,
            names=list(d.keys()),
            non_sampling_parameters=non_sampling_parameters,
        )
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


def dataframe_to_live_points(df, non_sampling_parameters=True):
    """Convert and pandas dataframe to live points.

    Adds the non-sampling parameter initialised to their defaults.

    Based on this answer on Stack Exchange:
    https://stackoverflow.com/a/51280608

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Pandas DataFrame to convert to live points
    non_sampling_parameters : bool
        Indicates whether non-sampling parameters should be included.

    Returns
    -------
    structured_array
        Numpy structured array with fields given by column names plus logP and
        logL.
    """
    if non_sampling_parameters:
        extra = config.livepoints.non_sampling_defaults
    else:
        extra = tuple()
    return np.array(
        [tuple(x) + extra for x in df.values],
        dtype=get_dtype(
            list(df.dtypes.index),
            non_sampling_parameters=non_sampling_parameters,
        ),
    )


def _unstructured_view_dtype(x, names):
    """Get the dtype for an unstructured view.

    Parameters
    ----------
    x : numpy.ndarray
        Array on which to base the dtype.
    names : Iterable
        Names for the fields to include in the dtype.

    Returns
    -------
    numpy.dtype
        Dtype containing the specified fields.
    """
    return np.dtype({name: x.dtype.fields[name] for name in names})


def unstructured_view(x, names=None, dtype=None):
    """Get an unstructured view of a live points containing certain parameters.

    This is quicker than converting to a unstructured array and does not
    create a copy of the array.

    Parameters
    ----------
    x : numpy.ndarray
        Structured array.
    names : Optional[Iterable]
        Iterable of parameters to include in the view. Must be specified if
        dtype is None.
    dtype : Optional[numpy.dtype]
        Dtype for constructing the unstructured view.

    Returns
    -------
    numpy.ndarray
        View of x as an unstructured array that contains only the
        parameters in names/dtype. Shape is (x.size, # parameters).
    """
    if dtype is None:
        dtype = _unstructured_view_dtype(x, names)
    return np.ndarray(x.shape, dtype, x, 0, x.strides).view(
        (config.livepoints.default_float_dtype, len(dtype))
    )
