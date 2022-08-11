# -*- coding: utf-8 -*-
"""
Utilities for computing distances.
"""
import numpy as np
from scipy.spatial import distance


def compute_minimum_distances(samples, metric="euclidean"):
    """
    Compute the distance to the nearest neighbour of each sample

    Parameters
    ----------
    samples : array_like
        Array of samples.
    metric : str, optional
        Metric to use. See scipy docs for list of metrics:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    Returns
    -------
    array_like
        Distance to nearest neighbour for each sample.
    """
    d = distance.cdist(samples, samples, metric)
    d[d == 0] = np.nan
    dmin = np.nanmin(d, axis=1)
    return dmin
