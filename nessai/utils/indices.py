# -*- coding: utf-8 -*-
"""
Utilities related to insertion indices.
"""
import numpy as np
from scipy import stats


def compute_indices_ks_test(indices, nlive, mode="D+"):
    """
    Compute the two-sided KS test for discrete insertion indices for a given
    number of live points

    Parameters
    ----------
    indices : array_like
        Indices of newly inserted live points
    nlive : int
        Number of live points

    Returns
    -------
    D : float
        Two-sided KS statistic
    p : float
        p-value
    """
    if len(indices):
        counts = np.zeros(nlive)
        u, c = np.unique(indices, return_counts=True)
        counts[u] = c
        cdf = np.cumsum(counts) / len(indices)
        if mode == "D+":
            D = np.max(np.arange(1.0, nlive + 1) / nlive - cdf)
        elif mode == "D-":
            D = np.max(cdf - np.arange(0.0, nlive) / nlive)
        else:
            raise RuntimeError(f"{mode} is not a valid mode. Choose D+ or D-")
        p = stats.ksone.sf(D, len(indices))
        return D, p
    else:
        return None, None


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply the Bonferroni correction for multiple tests.

    Based on the implementation in ``statmodels.stats.multitest``

    Parameters
    ----------
    p_values :  array_like
        Uncorrelated p-values.
    alpha : float, optional
        Family wise error rate.
    """
    p_values = np.asarray(p_values)
    alpha_bon = alpha / p_values.size
    reject = p_values <= alpha_bon
    p_values_corrected = p_values * p_values.size
    p_values_corrected[p_values_corrected > 1] = 1
    return reject, p_values_corrected, alpha_bon
