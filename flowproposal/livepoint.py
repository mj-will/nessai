import numpy as np
from numpy.lib import recfunctions as rfn


def get_dtype(names, array_dtype='f8'):
    """
    Get a list of the dtypes for the structed array
    """
    return [(n, array_dtype) for n in names] + [('logP', array_dtype),
            ('logL', 'f16')]

def live_points_to_array(live_points, names):
    """
    Converts live points to unstructered arrays for training
    """
    return rfn.structured_to_unstructured(live_points[names])


def parameters_to_live_point(parameters, names):
    """
    Take a list or array of parameters for a single live point
    and converts them to a live point
    """
    return np.array((*parameters, 0. , 0.), dtype=get_dtype(names, 'f8'))


def numpy_array_to_live_points(array, names):
    """
    Convert a numpy array to a numpy structure array with the
    correct fields
    """
    array = array.ravel().view(dtype=[(n, 'f8') for n in names])
    array = rfn.append_fields(array, ['logP', 'logL'], data=[*np.zeros([array.size, 2]).T],
                dtypes=['f8', 'f16'], usemask=False)
    return array


def dict_to_live_points(d):
    """
    Convert a dictionary with parameters names as keys to live points
    """
    N = len(list(d.values())[0])
    if N == 1:
        return np.array((*list(d.values()), 0. , 0.),
            dtype=get_dtype(d.keys(), 'f8'))
    else:
        array = np.zeros(N, dtype=[(n, 'f8') for n in d.keys()])
        for k, v in d.items():
            array[k] = v
        array = rfn.append_fields(array, ['logP', 'logL'], data=[*np.zeros([array.size, 2]).T],
                dtypes=['f8', 'f16'], usemask=False)
        return array

