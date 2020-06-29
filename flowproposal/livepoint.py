import numpy as np
from numpy.lib import recfunctions as rfn

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
    return np.array([(*parameters, 0. , 0.)],
            dtype=[(n, 'f') for n in names + ['logP', 'logL']])


def numpy_array_to_live_points(array, names):
    """
    Convert a numpy array to a numpy structure array with the
    correct fields
    """
    array = np.concatenate([array, np.zeros([array.shape[0], 2])], axis=-1).astype('float32')
    return array.ravel().view(
            dtype=[(n, 'f') for n in names + ['logP', 'logL']])


def dict_to_live_points(d):
    """
    Convert a dictionary with parameters names as keys to live points
    """
    array = np.zeros(len(list(d.values())[0]), dtype=[(n, 'f') for n in d.keys()])
    for k, v in d.items():
        array[k] = v
    array = rfn.append_fields(array, ['logP', 'logL'], data=[*np.zeros([array.size, 2]).T],
            dtypes=['f', 'f'], usemask=False)
    return array

