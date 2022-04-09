# -*- coding: utf-8 -*-
"""Utilities for computing information and entropy"""
import numpy as np
from scipy.special import logsumexp


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


def kl_divergence_from_log(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """KL Divergence where inputs are log-probabilties."""
    log_p = log_p - logsumexp(log_p)
    log_q = log_q - logsumexp(log_q)
    return np.mean(log_p - log_q)


def relative_entropy_from_log(
    log_p: np.ndarray, log_q: np.ndarray
) -> np.ndarray:
    """Relative entropy between samples from log-probabilities."""
    log_p = log_p - logsumexp(log_p)
    log_q = log_q - logsumexp(log_q)
    return log_p - log_q
