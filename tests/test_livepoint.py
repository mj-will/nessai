
import numpy as np
import flowproposal.livepoint as lp


def test_parameters_to_live_point():
    """
    Test function that produces a single live point given the parameter
    values for the live point as a live or an array
    """
    truth = np.array([(1., 2., 3., 0., 0.)], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('logP', 'f8'), ('logL', lp.logL_type)])
    live_point = lp.parameters_to_live_point([1., 2., 3.], ['x', 'y', 'z'])

    np.testing.assert_array_equal(truth, live_point)


def test_numpy_array_to_live_points():
    """
    Test the fuction the produces an array of live points given numpy array
    of shape [# point, # dimensions]
    """
    truth = np.array([(1., 2., 3., 0., 0.), (4., 5., 6., 0., 0.)],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('logP', 'f8'), ('logL', lp.logL_type)])
    array = np.array([[1., 2., 3.], [4., 5., 6.]])
    live_points = lp.numpy_array_to_live_points(array, names=['x', 'y', 'z'])

    np.testing.assert_array_equal(truth, live_points)


def test_dict_to_live_points():
    """
    Test the function that converts a dictionary with live points to
    a live point array
    """
    truth = np.array([(1., 2., 3., 0., 0.), (4., 5., 6., 0., 0.)],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('logP', 'f8'), ('logL', lp.logL_type)])
    d = {'x': [1, 4], 'y': [2, 5], 'z': [3, 6]}
    live_points = lp.dict_to_live_points(d)
    np.testing.assert_array_equal(truth, live_points)
    np.testing.assert_array_equal(truth, live_points)
