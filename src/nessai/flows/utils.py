# -*- coding: utf-8 -*-
"""
Various utilities for implementing normalising flows.
"""

import copy
import inspect
import logging
import warnings
from typing import Callable, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from glasflow.distributions import MultivariateUniform
from glasflow.nflows import transforms
from glasflow.nflows.distributions import Distribution

from .distributions import MultivariateNormal, ResampledGaussian
from .nets import MLP

logger = logging.getLogger(__name__)


def silu(x):
    """
    SiLU (Sigmoid-weighted Linear Unit) activation function.

    Also known as swish.

    Elfwing et al 2017: https://arxiv.org/abs/1702.03118v3
    """
    return torch.mul(x, torch.sigmoid(x))


def get_base_distribution(
    n_inputs: int, distribution: Union[str, Type[Distribution]], **kwargs
) -> Distribution:
    """Get the base distribution for a flow.

    Includes special configuration for certain distributions.

    Parameters
    ----------
    n_inputs : int
        Number of inputs to the distribution.
    distribution : Union[str, Type[glasflow.nflows.distribution.Distribution]]
        Distribution class or name of known distribution
    kwargs : Any
        Keyword arguments used when creating an instance of distribution.
    """
    distributions = {
        "mvn": MultivariateNormal,
        "normal": MultivariateNormal,
        "lars": ResampledGaussian,
        "resampled": ResampledGaussian,
        "uniform": MultivariateUniform,
    }

    DistClass = None

    if isinstance(distribution, str):
        DistClass = distributions.get(distribution.lower())
        if not DistClass:
            raise ValueError(f"Unknown distribution: {distribution}")
    elif inspect.isclass(distribution):
        logger.debug("Distribution is class. Creating an instance.")
        DistClass = distribution

    if DistClass:
        logger.debug("Creating instance of the base distribution")
        if DistClass is ResampledGaussian:
            n_layers = kwargs.pop("n_layers", 2)
            n_neurons = get_n_neurons(
                kwargs.pop("n_neurons", None), n_inputs=n_inputs
            )
            layers_list = n_layers * [n_neurons]
            logger.debug(
                f"LARS acceptance network will have {n_layers} layers with "
                f"{n_neurons} neurons each."
            )
            net_kwargs = kwargs.pop("net_kwargs", {})
            acc_fn = MLP(
                [n_inputs],
                [1],
                layers_list,
                activate_output=torch.sigmoid,
                **net_kwargs,
            )
            logger.debug(f"Other LARs kwargs: {kwargs}")
            dist = DistClass([n_inputs], acc_fn, **kwargs)
        elif DistClass is MultivariateUniform:
            dist = DistClass(
                low=torch.zeros(n_inputs, dtype=torch.get_default_dtype()),
                high=torch.ones(n_inputs, dtype=torch.get_default_dtype()),
            )
        else:
            dist = DistClass([n_inputs], **kwargs)
    elif distribution is None:
        dist = None
    else:
        dist = distribution
    return dist


def get_n_neurons(
    n_neurons: Optional[int] = None,
    n_inputs: Optional[int] = None,
    default: int = 8,
) -> int:
    """Get the number of neurons.

    Notes
    -----
    If :code:`n_inputs` is also specified then the options for
    :code:`n_neurons` are either a value that can be converted to an
    :code:`int` or one of the following:

        - :code:`'auto'` or :code:`'double'`: uses twice the number of inputs
        - :code:`'equal'`: uses the number of inputs
        - :code:`'half'`: uses half the number of inputs
        - :code:`None`: falls back to :code:`'auto'`

    Parameters
    ----------
    n_neurons : Optional[int]
        Number of neurons.
    n_inputs: Optional[int]
        Number of inputs.
    default : int
        Default value if :code:`n_neurons` and :code:`n_inputs` are not given.

    Returns
    -------
    int
        Number of neurons.

    Raises
    ------
    ValueError
        Raised if the number of inputs could not be converted to an integer.
    """
    if n_inputs is None:
        if n_neurons is None:
            n = default
        else:
            n = n_neurons
    else:
        if n_neurons is None or n_neurons in {"auto", "double"}:
            n = 2 * n_inputs
        elif n_neurons == "equal":
            n = n_inputs
        elif n_neurons == "half":
            n = n_inputs // 2
        else:
            n = n_neurons
    try:
        n = int(n)
    except ValueError:
        raise ValueError(
            "Could not get number of neurons. `n_neurons` was set to "
            f"`{n_neurons}` which could not be translated to a valid int. If "
            "using `auto`, `double`, `equal` or `half`, check that `n_inputs` "
            "is set."
        )
    return n


def get_native_flow_class(name):
    """Get a natively implemented flow class."""
    name = name.lower()
    from .maf import MaskedAutoregressiveFlow
    from .nsf import NeuralSplineFlow
    from .realnvp import RealNVP

    flows = {
        "realnvp": RealNVP,
        "maf": MaskedAutoregressiveFlow,
        "frealnvp": RealNVP,
        "spline": NeuralSplineFlow,
        "nsf": NeuralSplineFlow,
    }
    if name not in flows:
        raise ValueError(f"Unknown flow: {name}")
    return flows.get(name)


def get_flow_class(name: str):
    """Get the class to use for the normalizing flow from a string."""
    name = name.lower()
    if "glasflow" in name:
        from ..experimental.flows.glasflow import get_glasflow_class

        logger.warning("Using experimental glasflow flow!")
        FlowClass = get_glasflow_class(name)
    else:
        FlowClass = get_native_flow_class(name)
    return FlowClass


def get_activation_function(name: str) -> Callable:
    """Get the activation function from a string."""
    activations = {"relu": F.relu, "tanh": F.tanh, "swish": silu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def configure_model(config):
    """
    Setup the flow form a configuration dictionary.
    """
    config = copy.deepcopy(config)

    if not isinstance(config["n_inputs"], int):
        raise TypeError("Number of inputs (n_inputs) must be an int")

    kwargs_dict = config.pop("kwargs", None)
    if kwargs_dict is not None:
        warnings.warn(
            "Specifying the kwargs as a dictionary is deprecated.",
            FutureWarning,
        )
        config.update(kwargs_dict)

    if "activation" in config:
        config["activation"] = get_activation_function(config["activation"])

    dist_kwargs = config.pop("distribution_kwargs", {})
    if dist_kwargs is None:
        dist_kwargs = {}
    distribution = get_base_distribution(
        config["n_inputs"],
        config.pop("distribution", None),
        **dist_kwargs,
    )
    if distribution is not None:
        config["distribution"] = distribution

    FlowClass = config.pop("flow", None)
    ftype = config.pop("ftype", None)
    if FlowClass is None and ftype is None:
        raise RuntimeError("Must specify either 'flow' or 'ftype'.")

    if FlowClass is None:
        FlowClass = get_flow_class(ftype)

    model = FlowClass(
        config.pop("n_inputs"),
        config.pop("n_neurons"),
        config.pop("n_blocks"),
        config.pop("n_layers"),
        **config,
    )
    return model


def reset_weights(module):
    """Reset parameters of a given module in place.

    Uses the ``reset_parameters`` method from ``torch.nn.Module``

    Also checks the following modules from glasflow.nflows

    - glasflow.nflows.transforms.normalization.BatchNorm

    Parameters
    ----------
    module : :obj:`torch.nn.Module`
        Module to reset
    """
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif isinstance(module, transforms.BatchNorm):
        # glasflow.nflows BatchNorm does not have a weight reset, so must
        # be done manually
        constant = np.log(np.exp(1 - module.eps) - 1)
        module.unconstrained_weight.data.fill_(constant)
        module.bias.data.zero_()
        module.running_mean.zero_()
        module.running_var.fill_(1)
    else:
        logger.warning(f"Could not reset: {module}")


def reset_permutations(module):
    """Resets permutations and linear transforms for a given module in place.

    Resets using the original initialisation method. This needed since they
    do not have a ``reset_parameters`` method.

    Parameters
    ----------
    module : :obj:`torch.nn.Module`
        Module to reset
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from .transforms import LULinear
    if isinstance(module, (transforms.LULinear, LULinear)):
        module.cache.invalidate()
        module._initialize(identity_init=True)
    elif isinstance(module, transforms.RandomPermutation):
        module._permutation = torch.randperm(len(module._permutation))


def create_linear_transform(linear_transform, features):
    """Function for creating linear transforms.

    Parameters
    ----------
    linear_transform : {'permutation', 'lu', 'svd'}
        Linear transform to use.
    featres : int
        Number of features.
    """
    if linear_transform.lower() == "permutation":
        return transforms.RandomPermutation(features=features)
    elif linear_transform.lower() == "lu":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.LULinear(
                    features, identity_init=True, using_cache=True
                ),
            ]
        )
    elif linear_transform.lower() == "svd":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.SVDLinear(
                    features, num_householder=10, identity_init=True
                ),
            ]
        )
    else:
        raise ValueError(
            f"Unknown linear transform: {linear_transform}. "
            "Choose from: {permutation, lu, svd}."
        )


def create_pre_transform(pre_transform, features, **kwargs):
    """Create a pre transform.

    Parameters
    ----------
    pre_transform : str, {logit, batch_norm}
        Name of the transform
    features : int
        Number of input features
    kwargs :
        Keyword arguments passed to the transform class.
    """
    if pre_transform == "logit":
        return transforms.Logit(**kwargs)
    elif pre_transform == "batch_norm":
        return transforms.BatchNorm(features=features, **kwargs)
    else:
        raise ValueError(f"Unknown pre-transform: {pre_transform}")
