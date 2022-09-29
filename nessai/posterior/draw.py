# -*- coding: utf-8 -*-
"""
Functions for drawing posterior samples
"""
import logging

import numpy as np
from scipy.special import logsumexp

from ..utils.stats import effective_sample_size
from .weights import compute_weights

logger = logging.getLogger(__name__)


def draw_posterior_samples(
    nested_samples,
    nlive=None,
    n=None,
    log_w=None,
    method="rejection_sampling",
    return_indices=False,
):
    """Draw posterior samples given the nested samples.

    Requires either the posterior weights or the number of live points.

    Parameters
    ----------
    nested_samples : structured array
        Array of nested samples.
    nlive : int, optional
        Number of live points used during nested sampling. Either this
        arguments or log_w must be specified.
    n : int, optional
        Number of samples to draw. Only used for importance sampling. If not
        specified, the effective sample size is used instead.
    log_w : array_like, optional
        Array of posterior weights. If specified the weights are not computed
        and these weights are used instead.
    method : str
        Method for drawing the posterior samples. Choose from

            - :code:`'rejection_sampling'`
            - :code:`'multinomial_resampling'`
            - :code:`'importance_sampling'` (same as multinomial)
    return_indices : bool
        If true return the indices of the accepted samples.

    Returns
    -------
    posterior_samples : numpy.ndarray
        Samples from the posterior distribution.
    indices : numpy.ndarray
        Indices of the accepted samples in the original nested samples.
        Only returned if :code:`return_indices` is True.

    Raises
    ------
    ValueError
        If the chosen method is not a valid method.
    """
    nested_samples = np.asarray(nested_samples)
    if log_w is None:
        _, log_w = compute_weights(nested_samples["logL"], nlive)
    else:
        log_w = np.asarray(log_w)
    ess = effective_sample_size(log_w)
    logger.info(f"Effective sample size: {ess:.1f}")
    if method == "rejection_sampling":
        logger.info("Producing posterior samples using rejection sampling")
        if n is not None:
            logger.warning(
                "Number of samples cannot be specified for rejection sampling"
            )
        log_w = log_w - np.max(log_w)
        log_u = np.log(np.random.rand(nested_samples.size))
        indices = np.where(log_w > log_u)[0]
        samples = nested_samples[indices]
    elif method in ["importance_sampling", "multinomial_resampling"]:
        logger.info("Producing posterior samples using multinomial resampling")
        if n is None:
            n = int(ess)
        log_w = log_w - logsumexp(log_w)
        indices = np.random.choice(
            nested_samples.size, size=n, p=np.exp(log_w), replace=True
        )
        samples = nested_samples[indices]
    else:
        raise ValueError(
            f"Unknown method of drawing posterior samples: {method}"
        )

    if return_indices:
        return samples, indices
    else:
        return samples
