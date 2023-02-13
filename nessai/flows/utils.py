# -*- coding: utf-8 -*-
"""
Various utilities for implementing normalising flows.
"""
import inspect
import logging
from typing import Optional, Type, Union
import warnings

from glasflow.nflows import transforms
from glasflow.nflows.distributions import Distribution
import numpy as np
import torch
import torch.nn.functional as F

from .distributions import MultivariateNormal


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


def configure_model(config):
    """
    Setup the flow form a configuration dictionary.
    """
    from .realnvp import RealNVP
    from .maf import MaskedAutoregressiveFlow
    from .nsf import NeuralSplineFlow
    from ..flowmodel import config as fmconfig

    kwargs = {}
    flows = {
        "realnvp": RealNVP,
        "maf": MaskedAutoregressiveFlow,
        "frealnvp": RealNVP,
        "spline": NeuralSplineFlow,
        "nsf": NeuralSplineFlow,
    }
    activations = {"relu": F.relu, "tanh": F.tanh, "swish": silu, "silu": silu}

    config = config.copy()

    if not isinstance(config["n_inputs"], int):
        raise TypeError("Number of inputs (n_inputs) must be an int")

    allowed_keys = set(fmconfig.DEFAULT_MODEL_CONFIG.keys())
    extra_keys = set(config.keys()) - allowed_keys
    if extra_keys:
        raise RuntimeError(
            f"Unknown keys in model config: {extra_keys}. "
            f"Known keys are: {allowed_keys}"
        )

    k = config.get("kwargs", None)
    if k is not None:
        if "activation" in k and isinstance(k["activation"], str):
            try:
                k["activation"] = activations[k["activation"]]
            except KeyError as e:
                raise RuntimeError(f"Unknown activation function: {e}")

        kwargs.update(k)

    dist_kwargs = config.pop("distribution_kwargs", None)
    if dist_kwargs is None:
        dist_kwargs = {}
    distribution = get_base_distribution(
        config["n_inputs"],
        config.pop("distribution", None),
        **dist_kwargs,
    )
    if distribution:
        kwargs["distribution"] = distribution

    fc = config.get("flow", None)
    ftype = config.get("ftype", None)
    if fc is not None:
        model = fc(
            config["n_inputs"],
            config["n_neurons"],
            config["n_blocks"],
            config["n_layers"],
            **kwargs,
        )
    elif ftype is not None:
        if ftype.lower() not in flows:
            raise RuntimeError(
                f"Unknown flow type: {ftype}. Choose from:" f"{flows.keys()}"
            )
        if (
            ("mask" in kwargs and kwargs["mask"] is not None)
            or ("net" in kwargs and kwargs["net"] is not None)
        ) and ftype.lower() not in ["realnvp", "frealnvp"]:
            raise RuntimeError(
                "Custom masks and networks are only " "supported for RealNVP"
            )

        model = flows[ftype.lower()](
            config["n_inputs"],
            config["n_neurons"],
            config["n_blocks"],
            config["n_layers"],
            **kwargs,
        )
    else:
        raise RuntimeError("Must specify either 'flow' or 'ftype'.")

    device = torch.device(config.get("device_tag", "cpu"))
    if device != "cpu":
        try:
            model.to(device)
        except RuntimeError as e:
            device = torch.device("cpu")
            logger.warning(
                "Could not send the normalising flow to the "
                f"specified device {config['device_tag']} send to CPU "
                f"instead. Error raised: {e}"
            )
    logger.debug("Flow model:")
    logger.debug(model)

    model.device = device

    return model, device


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
