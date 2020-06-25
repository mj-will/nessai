
import numpy as np
from flowproposal.livepoint import parameters_to_live_point, numpy_array_to_live_points


def test_parameters_to_live_point():
    """
    Test function that produces a single live point given the parameter
    values for the live point as a live or an array
    """
    truth = np.array([(1., 2., 3., 0., 0.)], dtype=[('x', 'f'), ('y', 'f'), ('z', 'f'),
        ('logP', 'f'), ('logL', 'f')])
    live_point = parameters_to_live_point([1., 2., 3.], ['x', 'y', 'z'])

    np.testing.assert_array_equal(truth, live_point)


def test_numpy_array_to_live_points():
    """
    Test the fuction the produces an array of live points given numpy array
    of shape [# point, # dimensions]
    """
    truth = np.array([(1., 2., 3., 0., 0.), (4., 5., 6., 0., 0.)],
            dtype=[('x', 'f'), ('y', 'f'), ('z', 'f'), ('logP', 'f'), ('logL', 'f')])
    array = np.array([[1., 2., 3.], [4., 5., 6.]])
    live_points = numpy_array_to_live_points(array, names=['x', 'y', 'z'])

    np.testing.assert_array_equal(truth, live_points)

