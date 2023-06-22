# -*- coding: utf-8 -*-
"""Utilities for computing information and entropy"""
import numpy as np


def differential_entropy(log_p: np.ndarray) -> float:
    """Approximate the differential entropy from samples.

    Notes
    -----
    Assumes samples are drawn from :math:`p(x)` such that

    .. math::
        h(x) = -\\int p(x) \\log p(x) dx

    can be approximated via Monte Carlo integration.

    Parameters
    ----------
    log_p : numpy.ndarray
        Array of log-probabilities.

    Returns
    -------
    float
        The differential entropy
    """
    return -np.mean(log_p)
