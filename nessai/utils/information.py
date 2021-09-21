# -*- coding: utf-8 -*-
"""Utitilies for computing information and entropy"""
import numpy as np


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
        Arrray of probabilities
    """
    return self_entropy(p) / np.log(len(p))
