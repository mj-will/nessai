import pytest
import numpy as np

def test_new_point(model):
    """
    Test the default method for generating a new point with the bounds
    """
    new_point = model.new_point()
    assert (new_point['x'] < 5) & (new_point['y'] > -5)
    assert (new_point['y'] < 5) & (new_point['y'] > -5)
