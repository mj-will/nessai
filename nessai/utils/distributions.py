# -*- coding: utf-8 -*-
"""
Utilities for getting distributions for use in \
        :py:class:`~nessai.proposal.flowproposal.FlowProposal`
"""
from glasflow.nflows.distributions.uniform import BoxUniform
import torch
from torch.distributions import MultivariateNormal


def get_uniform_distribution(dims, r, device="cpu"):
    """
    Return a torch distribution that is uniform in the number of dims
    specified.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    r : float
        Radius to use for lower and upper bounds.
    device : str, optional
        Device on which the distribution is placed.

    Returns
    -------
    :obj:`glasflow.nflows.distributions.uniform.BoxUniform`
        Instance of BoxUniform which the lower and upper bounds set by
        the radius
    """
    r = r * torch.ones(dims, device=device)
    return BoxUniform(low=-r, high=r)


def get_multivariate_normal(dims, var=1, device="cpu"):
    """
    Return a Pytorch distribution that is normally distributed in n dims
    with a given variance.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    var : float, optional
        Variance.
    device : str, optional
        Device on which the distribution is placed.

    Returns
    -------
    :obj:`nessai.flows.distributions.MultivariateNormal`
        Instance of MultivariateNormal with correct variance and dims.
    """
    loc = torch.zeros(dims).to(device).double()
    covar = var * torch.eye(dims).to(device).double()
    return MultivariateNormal(loc, covariance_matrix=covar)
