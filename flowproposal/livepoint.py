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


class OrderedLivePoints:

    def __init__(self, live_points):
        """
        Initlaise the ordered live points.
        """
        self.live_points = np.sort(live_points, order='logL')

    def insert_live_point(self, live_point):
        """
        Insert a live point
        """
        # This is the index including the current worst point, so final index
        # is one less, otherwise index=0 would never be possible
        index = np.searchsorted(self.live_points['logL'], live_point['logL'])
        # Concatentate is complied C code, so it is much faster than np.insert
        # it also allows for simultaneous removal of the worst point
        # and insertion of the new live point
        self.live_points = np.concatenate([self.live_points[1:i], live_point,
            self.live_points[i:]])
        return index - 1

    def tolist(self):
        """
        Convert the set of live points to a list
        """
        return self.live_points.tolist()
