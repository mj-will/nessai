# -*- coding: utf-8 -*-
"""Utitilies for computing information and entropy"""
import numpy as np
from scipy.special import logsumexp


def self_entropy(p: np.ndarray) -> float:
    """Compute the self entropy of an array of probabilties.

    Parameters
    ----------
    p : array_like
        Array of probabilities.
    """
    p = np.array(p)
    p /= np.sum(p)
    return - np.sum(p * np.log(p))


def efficiency(p: np.ndarray) -> float:
    """Compute the efficiency (normalised entropy) for an array of \
        probabilities.

    Parameters
    ----------
    p : array_like
        Array of probabilities
    """
    return self_entropy(p) / np.log(len(p))


def cumulative_entropy(p: np.ndarray, base: float = np.e) -> np.ndarray:
    """Compute the entropy assuming a fraction of the points are discarded.

    Starts with a single point and adds a point each time until all
    of the points have been added.
    """
    assert p.size <= 10_000
    p = np.array(p, dtype=np.float128)
    norm = np.cumsum(p)
    q = p / norm[np.newaxis, :].T
    q[np.triu_indices(p.size, k=1)] = np.nan
    h = -np.nansum(q * np.log(q) / np.log(base), axis=1)
    return h


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
