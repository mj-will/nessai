# -*- coding: utf-8 -*-
"""
Utilities related to optimisation.
"""
import logging
from typing import Optional

import numpy as np
import numpy.lib.recfunctions as rfn
from functools import partial
from scipy.optimize import minimize
from scipy.special import logsumexp


logger = logging.getLogger(__name__)


def optimise_meta_proposal_weights(
    samples,
    method: str = "SLSQP",
    options: Optional[dict] = None,
    initial_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimise the weights of the meta proposal.

    Uses :code:`scipy.optimize.minimize`.

    Parameters
    ----------
    samples
        Samples drawn from the initial meta proposal.
    log_q
        Array of log probabilities for each proposal for each sample.
    method
        Optimisation method to use. See scipy docs for details.
    options
        Dictionary of options for :code:`scipy.optimize.minimize`.
    """
    if options is None and method == "SLSQP":
        options = dict(ftol=1e-10)

    qIDs = samples.log_q.dtype.names
    n_prop = len(qIDs)

    counts = np.unique(samples.samples["qID"], return_counts=True)[1]
    if initial_weights is None:
        initial_weights = counts / counts.sum()
    else:
        initial_weights = initial_weights / initial_weights.sum()

    log_p_hat = (
        samples.samples["logL"]
        + samples.samples["logW"]
        - samples.state.log_evidence
    )
    p_hat = np.exp(log_p_hat)

    def loss_fn(weights):
        """Computes the KL"""
        weights /= weights.sum()
        log_Q = rfn.apply_along_fields(
            partial(logsumexp, b=weights), samples.log_q
        )
        p_log_p = np.mean(p_hat * log_p_hat)
        p_log_q = np.mean(p_hat * log_Q)
        # print(p_log_p, p_log_q)
        return p_log_p - p_log_q

    # Weights must sum to one
    constraint = {"type": "eq", "fun": lambda x: 1 - x.sum()}

    logger.info(
        f"Starting optimisation, initial loss={loss_fn(initial_weights)}"
    )
    logger.info(f"Initial weights:\n {initial_weights}")
    result = minimize(
        loss_fn,
        initial_weights,
        constraints=constraint,
        bounds=n_prop * [(0, 1)],
        method=method,
        options=options,
    )
    logger.info(f"Finished optimisation, final loss={result.fun}")
    logger.info(f"Final weights:\n {result.x}")
    weights = dict(zip(qIDs, result.x))
    return weights
