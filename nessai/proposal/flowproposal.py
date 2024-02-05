# -*- coding: utf-8 -*-
"""
Main proposal object that includes normalising flows.
"""
import copy
import datetime
from functools import partial
import logging
import os
import re
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.special import logsumexp
import torch

from .. import config
from ..flowmodel import FlowModel
from ..livepoint import (
    live_points_to_array,
    numpy_array_to_live_points,
    get_dtype,
    empty_structured_array,
)
from ..reparameterisations import (
    CombinedReparameterisation,
    get_reparameterisation,
)
from ..plot import plot_live_points, plot_1d_comparison, nessai_style
from .rejection import RejectionProposal
from ..utils import (
    compute_radius,
    get_uniform_distribution,
    save_live_points,
)
from ..utils.sampling import NDimensionalTruncatedGaussian
from ..utils.structures import get_subset_arrays

logger = logging.getLogger(__name__)


class FlowProposal(RejectionProposal):
    """
    Object that handles training and proposal points

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User defined model.
    flow_config : dict, optional
        Configuration for training the normalising flow. If None, uses default
        settings. Defaults to None.
    output : str, optional
        Path to output directory.
    plot : {True, False, 'all', 'min'}, optional
        Controls the plotting level: ``True`` - all plots; ``False`` - no
        plots; ``'all'`` -  all plots and ``'min'`` -  1d plots and loss.
    latent_prior : {'truncated_gaussian', 'gaussian', 'uniform_nsphere', \
            'gaussian'}, optional
        Prior distribution in the latent space. Defaults to
        'truncated_gaussian'.
    poolsize : int, optional
        Size of the proposal pool. Defaults to 10000.
    update_poolsize : bool, optional
        If True the poolsize is updated using the current acceptance of the
        nested sampler.
    max_poolsize_scale : int, optional
        Maximum scale for increasing the poolsize. E.g. if this value is 10
        and the poolsize is 1000 the maximum number of points in the pool
        is 10,000.
    drawsize : int, optional
        Number of points to simultaneously draw when populating the proposal
        Defaults to 10000
    check_acceptance : bool, optional
        If True the acceptance is computed after populating the pool. This
        includes computing the likelihood for every point. Default False.
    min_radius : float, optional
        Minimum radius used for population. If not specified not minimum is
        used.
    max_radius : float, optional
        If a float then this value is used as an upper limit for the
        computed radius when populating the proposal. If unspecified no
        upper limit is used.
    fixed_radius : float, optional
        If specified and the chosen latent distribution is compatible, this
        radius will be used to draw new samples instead of the value computed
        with the flow.
    constant_volume_mode : bool
        If True, then a constant volume is used for the latent contour used to
        draw new samples. The exact volume can be set using `volume_fraction`
    volume_fraction : float
        Fraction of the total probability to contain with the latent contour
        when using a constant volume.
    compute_radius_with_all : bool, optional
        If True all the radius of the latent contour is computed using the
        maximum radius of all the samples used to train the flow.
    fuzz : float, optional
        Fuzz-factor applied to the radius. If unspecified no fuzz-factor is
        applied.
    expansion_fraction : float, optional
        Similar to ``fuzz`` but instead a scaling factor applied to the radius
        this specifies a rescaling for volume of the n-ball used to draw
        samples. This is translated to a value for ``fuzz``.
    truncate_log_q : bool, optional
        Truncate proposals using minimum log-probability of the training data.
    rescale_parameters : list or bool, optional
        If True live points are rescaled to `rescale_bounds` before training.
        If an instance of `list` then must contain names of parameters to
        rescale. If False no rescaling is applied.
    rescale_bounds : list, optional
        Lower and upper bound to use for rescaling. Defaults to [-1, 1]. See
        `rescale_parameters`.
    update_bounds : bool, optional
        If True bounds used for rescaling are updated at the starting of
        training. If False prior bounds are used.
    boundary_inversion : bool or list, optional
        If True boundary inversion is applied to all bounds. If
        If an instance of `list` of parameters names, then inversion
        only applied to these parameters. If False (default )no inversion is
        used.
    inversion_type : {'split', 'duplicate'}
        Type of inversion to use. ``'split'`` keeps the number of samples
        the sample but mirrors half around the bound. ``'duplicate'`` mirrors
        all the samples at the bound.
    detect_edges : bool, optional
        If True, when applying the version the option of no inversion is
        allowed.
    detect_edges_kwargs : dict, optional
        Dictionary of keyword arguments passed to \
                :func:`nessai.utils.detect_edge`.
    reparameterisations : dict, optional
        Dictionary for configure more flexible reparameterisations. This
        ignores any of the other settings related to rescaling. For more
        details see the documentation.
    fallback_reparameterisation : None or str
        Name of the reparameterisation to be used for parameters that have not
        been specified in the reparameterisations dictionary. If None, the
        :py:class:`~nessai.reparameterisations.NullReparameterisation` is used.
        Reparameterisation should support multiple parameters.
    use_default_reparameterisations : bool, optional
        If True then reparameterisations will be used even if
        ``reparameterisations`` is None. The exact reparameterisations used
        will depend on
        :py:func:`~nessai.proposal.flowproposal.FlowProposal.add_default_reparameterisations`
        which may be overloaded by child classes. If not specified then the
        value of the attribute
        :py:attr:`~nessai.proposal.flowproposal.FlowProposal.use_default_reparameterisations`
        is used.
    reverse_reparameterisations : bool
        Passed to :code:`reverse_order` in
        :py:obj:`~nessai.reparameterisations.combined.CombinedReparameterisation`.
        Reverses the order of the reparameterisations.
    """

    use_default_reparameterisations = False
    """
    Indicates whether reparameterisations will be used be default in this
    class. Child classes can change this value a force the default
    behaviour to change without changing the keyword arguments.
    """

    def __init__(
        self,
        model,
        flow_config=None,
        output="./",
        poolsize=None,
        rescale_parameters=True,
        latent_prior="truncated_gaussian",
        constant_volume_mode=True,
        volume_fraction=0.95,
        fuzz=1.0,
        plot="min",
        fixed_radius=False,
        drawsize=None,
        check_acceptance=False,
        truncate_log_q=False,
        rescale_bounds=[-1, 1],
        expansion_fraction=4.0,
        boundary_inversion=False,
        inversion_type="split",
        update_bounds=True,
        min_radius=False,
        max_radius=50.0,
        max_poolsize_scale=10,
        update_poolsize=True,
        accumulate_weights=False,
        save_training_data=False,
        compute_radius_with_all=False,
        detect_edges=False,
        detect_edges_kwargs=None,
        reparameterisations=None,
        fallback_reparameterisation=None,
        use_default_reparameterisations=None,
        reverse_reparameterisations=False,
    ):

        super(FlowProposal, self).__init__(model)
        logger.debug("Initialising FlowProposal")

        self._x_dtype = False
        self._x_prime_dtype = False
        self._draw_func = None
        self._populate_dist = None

        self.flow = None
        self._flow_config = None
        self.populated = False
        self.populating = False  # Flag used for resuming during population
        self.indices = []
        self.training_count = 0
        self.populated_count = 0
        self.names = []
        self.training_data = None
        self.save_training_data = save_training_data
        self.x = None
        self.samples = None
        self.rescaled_names = []
        self.acceptance = []
        self._edges = {}
        self._reparameterisation = None
        self.rescaling_set = False
        self.use_x_prime_prior = False
        self.accumulate_weights = accumulate_weights

        self.reparameterisations = reparameterisations
        if use_default_reparameterisations is not None:
            self.use_default_reparameterisations = (
                use_default_reparameterisations
            )
        self.fallback_reparameterisation = fallback_reparameterisation
        self.reverse_reparameterisations = reverse_reparameterisations

        self.output = output

        self.configure_population(
            poolsize,
            drawsize,
            update_poolsize,
            max_poolsize_scale,
            fuzz,
            expansion_fraction,
            latent_prior,
        )

        self.rescale_parameters = rescale_parameters
        self.update_bounds = update_bounds
        self.check_acceptance = check_acceptance
        self.rescale_bounds = rescale_bounds
        self.truncate_log_q = truncate_log_q
        self.boundary_inversion = boundary_inversion
        self.inversion_type = inversion_type
        self.flow_config = flow_config
        self.constant_volume_mode = constant_volume_mode
        self.volume_fraction = volume_fraction

        self.detect_edges = detect_edges
        self.detect_edges_kwargs = detect_edges_kwargs

        self.compute_radius_with_all = compute_radius_with_all
        self.configure_fixed_radius(fixed_radius)
        self.configure_min_max_radius(min_radius, max_radius)

        self.configure_plotting(plot)

        self.configure_latent_prior()
        self.alt_dist = None

    @property
    def poolsize(self):
        """
        Return the poolsize based of the base value and the current
        value of the scaling
        """
        return int(self._poolsize_scale * self._poolsize)

    @property
    def flow_config(self):
        """Return the configuration for the flow"""
        return self._flow_config

    @flow_config.setter
    def flow_config(self, config):
        """Set configuration.

        Does not update with the defaults because the number of inputs
        has not been determined yet.
        """
        if config is None:
            config = {}
        if "model_config" not in config:
            config["model_config"] = {}
        self._flow_config = config

    @property
    def dims(self):
        """Return the number of dimensions"""
        return len(self.names)

    @property
    def rescaled_dims(self):
        """Return the number of rescaled dimensions"""
        return len(self.rescaled_names)

    @property
    def x_dtype(self):
        """Return the dtype for the x space"""
        if not self._x_dtype:
            self._x_dtype = get_dtype(
                self.names, config.livepoints.default_float_dtype
            )
        return self._x_dtype

    @property
    def x_prime_dtype(self):
        """Return the dtype for the x prime space"""
        if not self._x_prime_dtype:
            self._x_prime_dtype = get_dtype(
                self.rescaled_names, config.livepoints.default_float_dtype
            )
        return self._x_prime_dtype

    @property
    def population_dtype(self):
        """
        dtype used for populating the proposal, depends on if the prior
        is defined in the x space or x-prime space
        """
        if self.use_x_prime_prior:
            return self.x_prime_dtype
        else:
            return self.x_dtype

    def configure_population(
        self,
        poolsize,
        drawsize,
        update_poolsize,
        max_poolsize_scale,
        fuzz,
        expansion_fraction,
        latent_prior,
    ):
        """
        Configure settings related to population
        """
        if poolsize is None:
            raise RuntimeError("Must specify a poolsize!")

        if drawsize is None:
            drawsize = poolsize

        self._poolsize = poolsize
        self._poolsize_scale = 1.0
        self.update_poolsize = update_poolsize
        self.max_poolsize_scale = max_poolsize_scale
        self.ns_acceptance = 1.0
        self.drawsize = drawsize
        self.fuzz = fuzz
        self.expansion_fraction = expansion_fraction
        self.latent_prior = latent_prior

    def configure_plotting(self, plot):
        """Configure plotting.

        Plotting is split into training and pool. Training refers to plots
        produced during training and pool refers to plots produces during
        the population stage.

        Parameters
        ----------
        plot : {True, False, 'all', 'train', 'pool', 'min', 'minimal'}
            Level of plotting. `all`, `train` and `pool` enable corner style
            plots. All other values that evaluate to True enable 1d histogram
            plots. False disables all plotting.
        """
        if plot:
            if isinstance(plot, str):
                if plot == "all":
                    self._plot_pool = "all"
                    self._plot_training = "all"
                elif plot == "train":
                    self._plot_pool = False
                    self._plot_training = "all"
                elif plot == "pool":
                    self._plot_pool = "all"
                    self._plot_training = False
                elif plot == "minimal" or plot == "min":
                    self._plot_pool = True
                    self._plot_training = True
                else:
                    logger.warning(
                        f"Unknown plot argument: {plot}, setting all false"
                    )
                    self._plot_pool = False
                    self._plot_training = False
            else:
                self._plot_pool = True
                self._plot_training = True

        else:
            self._plot_pool = False
            self._plot_training = False

    def configure_latent_prior(self):
        """Configure the latent prior"""
        if self.latent_prior == "truncated_gaussian":
            from ..utils import draw_truncated_gaussian

            self._draw_latent_prior = draw_truncated_gaussian

        elif self.latent_prior == "gaussian":
            logger.warning("Using a gaussian latent prior WITHOUT truncation")
            from ..utils import draw_gaussian

            self._draw_latent_prior = draw_gaussian
        elif self.latent_prior == "uniform":
            from ..utils import draw_uniform

            self._draw_latent_prior = draw_uniform
        elif self.latent_prior in ["uniform_nsphere", "uniform_nball"]:
            from ..utils import draw_nsphere

            self._draw_latent_prior = draw_nsphere
        elif self.latent_prior == "flow":
            self._draw_latent_prior = None
        else:
            raise RuntimeError(
                f"Unknown latent prior: {self.latent_prior}, choose from: "
                "truncated_gaussian (default), gaussian, "
                "uniform, uniform_nsphere"
            )

    def configure_fixed_radius(self, fixed_radius):
        """Configure the fixed radius"""
        if fixed_radius:
            try:
                self.fixed_radius = float(fixed_radius)
            except ValueError:
                logger.error(
                    "Fixed radius enabled but could not be converted to a "
                    "float. Setting fixed_radius=False"
                )
                self.fixed_radius = False
        else:
            self.fixed_radius = False

    def configure_min_max_radius(self, min_radius, max_radius):
        """
        Configure the minimum and maximum radius
        """
        if isinstance(min_radius, (int, float)):
            self.min_radius = float(min_radius)
        else:
            raise RuntimeError("Min radius must be an int or float")

        if max_radius:
            if isinstance(max_radius, (int, float)):
                self.max_radius = float(max_radius)
            else:
                raise RuntimeError("Max radius must be an int or float")
        else:
            logger.warning(
                "Running without a maximum radius! The proposal "
                "process may get stuck if very large radii are "
                "returned by the worst point."
            )
            self.max_radius = False

    def configure_constant_volume(self):
        """Configure using constant volume latent contour."""
        if self.constant_volume_mode:
            logger.debug("Configuring constant volume latent contour")
            if self.latent_prior == "truncated_gaussian":
                pass
            elif self.latent_prior in ["uniform_nball", "uniform_nsphere"]:
                logger.warning(
                    "Constant volume mode with latent_prior="
                    f"{self.latent_prior} is experimental!"
                )
            else:
                raise RuntimeError(
                    "Constant volume mode is not supported for latent_prior="
                    f"{self.latent_prior}"
                )
            self.fixed_radius = compute_radius(
                self.rescaled_dims, self.volume_fraction
            )
            self.fuzz = 1.0
            if self.max_radius < self.fixed_radius:
                logger.warning(
                    "Max radius is less than the radius need to use a "
                    "constant volume latent contour. Max radius will be "
                    "disabled."
                )
                self.max_radius = False
            if self.min_radius > self.fixed_radius:
                logger.warning(
                    "Min radius is greater than the radius need to use a "
                    "constant volume latent contour. Min radius will be "
                    "disabled."
                )
                self.min_radius = False
        else:
            logger.debug(
                "Nothing to configure for constant volume latent contour."
            )

    def update_flow_config(self):
        """Update the flow configuration dictionary."""
        self.flow_config["model_config"]["n_inputs"] = self.rescaled_dims

    def initialise(self):
        """
        Initialise the proposal class.

        This includes:
            * Setting up the rescaling
            * Verifying the rescaling is invertible
            * Initialising the FlowModel
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self._x_dtype = False
        self._x_prime_dtype = False

        self.set_rescaling()
        self.verify_rescaling()
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info("Overwriting fuzz factor with expansion fraction")
            self.fuzz = (1 + self.expansion_fraction) ** (
                1 / self.rescaled_dims
            )
            logger.info(f"New fuzz factor: {self.fuzz}")

        self.configure_constant_volume()
        self.update_flow_config()
        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def update_poolsize_scale(self, acceptance):
        """
        Update poolsize given the current acceptance.

        Parameters
        ----------
        acceptance : float
            Current acceptance.
        """
        logger.debug(f"Updating poolsize with acceptance: {acceptance:.3f}")
        if not acceptance:
            logger.warning("Acceptance is zero, using maximum scale")
            self._poolsize_scale = self.max_poolsize_scale
        else:
            self._poolsize_scale = 1.0 / acceptance
            if self._poolsize_scale > self.max_poolsize_scale:
                logger.debug("Poolsize scaling is greater than maximum value")
                self._poolsize_scale = self.max_poolsize_scale
            if self._poolsize_scale < 1.0:
                self._poolsize_scale = 1.0

    def add_default_reparameterisations(self):
        """Add any reparameterisations which are assumed by default"""
        logger.debug("No default reparameterisations")

    def get_reparameterisation(self, name):
        """Get the reparameterisation from the name"""
        return get_reparameterisation(name)

    def configure_reparameterisations(self, reparameterisations):
        """Configure the reparameterisations.

        Parameters
        ----------
        reparameterisations : {dict, None}
            Dictionary of reparameterisations. If None, then the defaults
            from :py:func`get_default_reparameterisations` are used.
        """
        if reparameterisations is None:
            logger.info(
                "No reparameterisations provided, using default "
                f"reparameterisations included in {self.__class__.__name__}"
            )
            _reparameterisations = {}
        else:
            _reparameterisations = copy.deepcopy(reparameterisations)
        logger.info(f"Adding reparameterisations from: {_reparameterisations}")
        self._reparameterisation = CombinedReparameterisation(
            reverse_order=self.reverse_reparameterisations
        )

        if not isinstance(_reparameterisations, dict):
            raise TypeError(
                "Reparameterisations must be a dictionary, "
                f"received {type(_reparameterisations).__name__}"
            )

        for k, cfg in _reparameterisations.items():
            if k in self.model.names:
                logger.debug(
                    f"Found parameter {k} in model, "
                    "assuming it is a parameter"
                )
                if isinstance(cfg, str) or cfg is None:
                    rc, default_config = self.get_reparameterisation(cfg)
                    default_config["parameters"] = k
                elif isinstance(cfg, dict):
                    if cfg.get("reparameterisation", None) is None:
                        raise RuntimeError(
                            f"No reparameterisation found for {k}. "
                            "Check inputs (and their spelling :)). "
                            f"Current keys: {list(cfg.keys())}"
                        )
                    rc, default_config = self.get_reparameterisation(
                        cfg["reparameterisation"]
                    )
                    cfg.pop("reparameterisation")

                    if cfg.get("parameters", False):
                        cfg["parameters"] += [k]
                    else:
                        default_config["parameters"] = k

                    default_config.update(cfg)
                else:
                    raise TypeError(
                        f"Unknown config type for: {k}. Expected str or dict, "
                        f"received instance of {type(cfg)}."
                    )
            else:
                logger.debug(f"Assuming {k} is a reparameterisation")
                try:
                    rc, default_config = self.get_reparameterisation(k)
                    default_config.update(cfg)
                    parameters = default_config.get("parameters")

                    if parameters is not None:
                        if not isinstance(parameters, list):
                            if isinstance(parameters, str):
                                patterns = [parameters]
                            else:
                                patterns = list(parameters)
                        else:
                            patterns = parameters.copy()
                        matches = []
                        for pattern in patterns:
                            r = re.compile(pattern)
                            matches += list(
                                filter(r.fullmatch, self.model.names)
                            )
                        default_config["parameters"] = matches
                    else:
                        logger.warning(
                            "Reparameterisation might be missing parameters!"
                        )

                except ValueError:
                    raise RuntimeError(
                        f"{k} is not a parameter in the model or a known "
                        "reparameterisation"
                    )

            if not default_config.get("parameters", False):
                raise RuntimeError(
                    "No parameters key in the config! "
                    "Check reparameterisations, setting logging"
                    " level to DEBUG can be helpful"
                )

            if (
                "boundary_inversion" in default_config
                and default_config["boundary_inversion"]
            ):
                self.boundary_inversion = True

            if isinstance(default_config["parameters"], list):
                prior_bounds = {
                    p: self.model.bounds[p]
                    for p in default_config["parameters"]
                }
            else:
                prior_bounds = {
                    default_config["parameters"]: self.model.bounds[
                        default_config["parameters"]
                    ]
                }

            logger.info(f"Adding {rc.__name__} with config: {default_config}")
            r = rc(prior_bounds=prior_bounds, **default_config)
            self._reparameterisation.add_reparameterisations(r)

        if self.use_default_reparameterisations:
            self.add_default_reparameterisations()

        other_params = [
            n
            for n in self.model.names
            if n not in self._reparameterisation.parameters
        ]
        if other_params:
            logger.debug("Getting fallback reparameterisation")
            FallbackClass, fallback_kwargs = self.get_reparameterisation(
                self.fallback_reparameterisation
            )
            fallback_kwargs["prior_bounds"] = {
                p: self.model.bounds[p] for p in other_params
            }
            logger.info(
                f"Assuming fallback reparameterisation "
                f"({FallbackClass.__name__}) for {other_params} with kwargs: "
                f"{fallback_kwargs}."
            )
            r = FallbackClass(parameters=other_params, **fallback_kwargs)
            self._reparameterisation.add_reparameterisations(r)

        if any(r._update_bounds for r in self._reparameterisation.values()):
            self.update_bounds = True
        else:
            self.update_bounds = False

        if self._reparameterisation.has_prime_prior:
            self.use_x_prime_prior = True
            self.x_prime_log_prior = self._reparameterisation.x_prime_log_prior
            logger.debug("Using x prime prior")
        else:
            logger.debug("Prime prior is disabled")
            if self._reparameterisation.requires_prime_prior:
                raise RuntimeError(
                    "One or more reparameterisations require use of the x "
                    "prime prior but it cannot be enabled with the current "
                    "settings."
                )

        self._reparameterisation.check_order()

        self.names = self._reparameterisation.parameters
        self.rescaled_names = self._reparameterisation.prime_parameters
        self.parameters_to_rescale = [
            p
            for p in self._reparameterisation.parameters
            if p not in self._reparameterisation.prime_parameters
        ]

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        if self.model.reparameterisations is not None:
            self.reparameterisations = self.model.reparameterisations
        elif (
            self.reparameterisations is not None
            or self.use_default_reparameterisations
        ):
            pass
        elif self.rescale_parameters:
            if not isinstance(self.rescale_parameters, list):
                self.rescale_parameters = self.model.names.copy()

            self.reparameterisations = {
                "rescaletobounds": {
                    "parameters": self.rescale_parameters,
                    "rescale_bounds": self.rescale_bounds,
                    "update_bounds": self.update_bounds,
                    "boundary_inversion": self.boundary_inversion,
                    "inversion_type": self.inversion_type,
                    "detect_edges": self.detect_edges,
                    "detect_edges_kwargs": self.detect_edges_kwargs,
                },
            }

        self.configure_reparameterisations(self.reparameterisations)

        logger.info(f"x space parameters: {self.names}")
        logger.info(f"parameters to rescale: {self.parameters_to_rescale}")
        logger.info(f"x prime space parameters: {self.rescaled_names}")
        self.rescaling_set = True

    def verify_rescaling(self):
        """
        Verify the rescaling functions are invertible
        """
        if not self.rescaling_set:
            raise RuntimeError(
                "Rescaling must be set before it can be verified"
            )
        logger.info("Verifying rescaling functions")
        x = self.model.new_point(N=1000)
        for inversion in ["lower", "upper", False, None]:
            self.check_state(x)
            logger.debug(f"Testing: {inversion}")
            x_prime, log_J = self.rescale(x, test=inversion)
            x_out, log_J_inv = self.inverse_rescale(x_prime)

            n = x.size
            ratio = x_out.size // x.size
            logger.debug(f"Ratio of output to input: {ratio}")
            for f in x.dtype.names:
                target = x[f]
                for count in range(ratio):
                    start = count * n
                    end = (count + 1) * n
                    block = x_out[f][start:end]
                    if f in config.livepoints.non_sampling_parameters:
                        if not np.allclose(block, target, equal_nan=True):
                            raise RuntimeError(
                                f"Non-sampling parameter {f} changed in "
                                f" the rescaling (block {count})."
                            )
                    elif not np.allclose(block, target, equal_nan=False):
                        raise RuntimeError(
                            f"Rescaling is not invertible for {f} "
                            f"(block {count})."
                        )
                    else:
                        logger.debug(f"Block {count} is equal to the input")
            if not np.allclose(log_J, -log_J_inv):
                raise RuntimeError("Rescaling Jacobian is not invertible")

        logger.info("Rescaling functions are invertible")

    def rescale(self, x, compute_radius=False, **kwargs):
        """
        Rescale from the physical space to the primed physical space

        Parameters
        ----------
        x: array_like
            Array of live points to rescale
        compute_radius: bool (False)
            Used to indicate when rescaling is being used for computing the
            radius for population. This is important for rescaling that uses
            inversions.

        Returns
        -------
        array
            Array of rescaled values
        array
            Array of log det|J|
        """
        x_prime = empty_structured_array(x.size, dtype=self.x_prime_dtype)
        log_J = np.zeros(x_prime.size)

        if x.size == 1:
            x = np.array([x], dtype=x.dtype)

        x, x_prime, log_J = self._reparameterisation.reparameterise(
            x, x_prime, log_J, compute_radius=compute_radius, **kwargs
        )

        for p in config.livepoints.non_sampling_parameters:
            x_prime[p] = x[p]
        return x_prime, log_J

    def inverse_rescale(self, x_prime, **kwargs):
        """
        Rescale from the primed physical space to the original physical
        space.

        Parameters
        ----------
        x_prime : array_like
            Array of live points to rescale.

        Returns
        -------
        array
            Array of rescaled values in the data space.
        array
            Array of log-Jacobian determinants.
        """
        x = empty_structured_array(x_prime.size, dtype=self.x_dtype)
        log_J = np.zeros(x.size)
        x, x_prime, log_J = self._reparameterisation.inverse_reparameterise(
            x, x_prime, log_J, **kwargs
        )

        for p in config.livepoints.non_sampling_parameters:
            x[p] = x_prime[p]
        return x, log_J

    def check_state(self, x):
        """Update the state of the proposal given some training data.

        Includes updating the reparameterisations.

        Parameters
        ----------
        x: array_like
            Array of training live points which can be used to set parameters
        """
        self._reparameterisation.update(x)

    @nessai_style()
    def _plot_training_data(self, output):
        """Plot the training data and compare to the results"""
        z_training_data, _ = self.forward_pass(
            self.training_data, rescale=True
        )
        z_gen = np.random.randn(self.training_data.size, self.dims)

        fig = plt.figure()
        plt.hist(np.sqrt(np.sum(z_training_data**2, axis=1)), "auto")
        plt.xlabel("Radius")
        fig.savefig(os.path.join(output, "radial_dist.png"))
        plt.close(fig)

        plot_1d_comparison(
            z_training_data,
            z_gen,
            labels=["z_live_points", "z_generated"],
            convert_to_live_points=True,
            filename=os.path.join(output, "z_comparison.png"),
        )

        x_prime_gen, log_prob = self.backward_pass(z_gen, rescale=False)
        x_prime_gen["logL"] = log_prob
        x_gen, log_J = self.inverse_rescale(x_prime_gen)
        x_gen, log_J, x_prime_gen = self.check_prior_bounds(
            x_gen, log_J, x_prime_gen
        )
        x_gen["logL"] += log_J

        plot_1d_comparison(
            self.training_data,
            x_gen,
            parameters=self.model.names,
            labels=["live points", "generated"],
            filename=os.path.join(output, "x_comparison.png"),
        )

        if self._plot_training == "all":
            plot_live_points(
                self.training_data,
                c="logL",
                filename=os.path.join(output, "x_samples.png"),
            )

            plot_live_points(
                x_gen,
                c="logL",
                filename=os.path.join(output, "x_generated.png"),
            )

        if self.parameters_to_rescale:
            if self._plot_training == "all":
                plot_live_points(
                    self.training_data_prime,
                    c="logL",
                    filename=os.path.join(output, "x_prime_samples.png"),
                )
                plot_live_points(
                    x_prime_gen,
                    c="logL",
                    filename=os.path.join(output, "x_prime_generated.png"),
                )

            plot_1d_comparison(
                self.training_data_prime,
                x_prime_gen,
                parameters=self.rescaled_names,
                labels=["live points", "generated"],
                filename=os.path.join(output, "x_prime_comparison.png"),
            )

    def train(self, x, plot=True):
        """
        Train the normalising flow given some of the live points.

        Parameters
        ----------
        x : structured_array
            Array of live points
        plot : {True, False, 'all'}
            Enable or disable plots for during training. By default the plots
            are only one-dimensional histograms, `'all'` includes corner
            plots with samples, these are often a few MB in size so
            proceed with caution!
        """
        if not self.initialised:
            raise RuntimeError("FlowProposal is not initialised.")

        if (plot and self._plot_training) or self.save_training_data:
            block_output = os.path.join(
                self.output, "training", f"block_{self.training_count}", ""
            )
        else:
            block_output = self.output

        if not os.path.exists(block_output):
            os.makedirs(block_output, exist_ok=True)

        if self.save_training_data:
            save_live_points(
                x, os.path.join(block_output, "training_data.json")
            )

        self.training_data = x.copy()
        self.check_state(self.training_data)

        x_prime, _ = self.rescale(x)

        self.training_data_prime = x_prime.copy()

        # Convert to numpy array for training and remove likelihoods and priors
        # Since the names of parameters may have changes, pull names from flows
        x_prime_array = live_points_to_array(
            x_prime, self.rescaled_names, copy=True
        )

        self.flow.train(
            x_prime_array,
            output=block_output,
            plot=self._plot_training and plot,
        )

        if self._plot_training and plot:
            self._plot_training_data(block_output)

        self.populated = False
        self.training_count += 1

    def reset_model_weights(self, **kwargs):
        """
        Reset the flow weights.

        Parameters
        ----------
        kwargs :
            Keyword arguments passed to
            :meth:`nessai.flowmodel.FlowModel.reset_model`.
        """
        self.flow.reset_model(**kwargs)

    def check_prior_bounds(self, x, *args):
        """
        Return only values that are within the prior bounds

        Parameters
        ----------
        x: array_like
            Array of live points which will compared against prior bounds
        *args:
            Additional arrays which correspond to the array of live points.
            Only those corresponding to points within the prior bounds
            are returned

        Returns
        -------
        out: tuple of arrays
            Array containing the subset of the original arrays which fall
            within the prior bounds
        """
        flags = self.model.in_bounds(x)
        return (a[flags] for a in (x,) + args)

    def forward_pass(self, x, rescale=True, compute_radius=True):
        """
        Pass a vector of points through the model

        Parameters
        ----------
        x : array_like
            Live points to map to the latent space
        rescale : bool, optional (True)
            Apply rescaling function
        compute_radius : bool, optional (True)
            Flag parsed to rescaling for rescaling specific to radius
            computation

        Returns
        -------
        x : array_like
            Samples in the latent space
        log_prob : array_like
            Log probabilities corresponding to each sample (including the
            jacobian)
        """
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x, compute_radius=compute_radius)
            log_J += log_J_rescale

        x = live_points_to_array(x, names=self.rescaled_names, copy=True)

        if x.ndim == 1:
            x = x[np.newaxis, :]
        z, log_prob = self.flow.forward_and_log_prob(x)

        return z, log_prob + log_J

    def backward_pass(
        self, z, rescale=True, discard_nans=True, return_z=False
    ):
        """
        A backwards pass from the model (latent -> real)

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space
        rescale : bool, optional (True)
            Apply inverse rescaling function
        discard_nan: bool
            If True, samples with NaNs or Infs in log_q are removed.
        return_z : bool
            If True, return the array of latent samples, this may differ from
            the input since samples can be discarded.

        Returns
        -------
        x : array_like
            Samples in the data space
        log_prob : array_like
            Log probabilities corresponding to each sample (including the
            Jacobian)
        z : array_like
            Samples in the latent space, only returned if :code:`return_z=True`
        """
        # Compute the log probability
        try:
            x, log_prob = self.flow.sample_and_log_prob(
                z=z, alt_dist=self.alt_dist
            )
        except AssertionError:
            return np.array([]), np.array([])

        if discard_nans:
            valid = np.isfinite(log_prob)
            x, log_prob = x[valid], log_prob[valid]
        x = numpy_array_to_live_points(
            x.astype(config.livepoints.default_float_dtype),
            self.rescaled_names,
        )
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_prob -= log_J
            x, z, log_prob = self.check_prior_bounds(x, z, log_prob)
        if return_z:
            return x, log_prob, z
        else:
            return x, log_prob

    def radius(self, z, *arrays):
        """
        Calculate the radius of a latent point or set of latent points.
        If multiple points are parsed the maximum radius is returned.

        Parameters
        ----------
        z : :obj:`np.ndarray`
            Array of points in the latent space
        *arrays :
            Additional arrays to return the corresponding value

        Returns
        -------
        tuple of arrays
            Tuple of array with the maximum radius and corresponding values
            from any additional arrays that were passed.
        """
        r = np.sqrt(np.sum(z**2.0, axis=-1))
        i = np.nanargmax(r)
        if arrays:
            return (r[i],) + tuple(a[i] for a in arrays)
        else:
            return r[i]

    def log_prior(self, x):
        """
        Compute the prior probability using the user-defined model

        Parameters
        ----------
        x : structured_array
            Array of samples

        Returns
        -------
        array_like
            Array of log prior probabilities
        """
        if self._reparameterisation:
            return self.model.batch_evaluate_log_prior(
                x
            ) + self._reparameterisation.log_prior(x)
        else:
            return self.model.batch_evaluate_log_prior(x)

    def x_prime_log_prior(self, x):
        """
        Compute the prior in the prime space

        Parameters
        ----------
        x : array
            Samples in the X-prime space.
        """
        raise RuntimeError("Prime prior is not implemented")

    def compute_weights(self, x, log_q, return_log_prior=False):
        """
        Compute weights for the samples.

        Does NOT normalise the weights

        Parameters
        ----------
        x :  structured_arrays
            Array of points
        log_q : array_like
            Array of log proposal probabilities.
        return_log_prior: bool
            If true, the log-prior probability is also returned.

        Returns
        -------
        array_like
            Log-weights for rejection sampling.
        """
        if self.use_x_prime_prior:
            log_p = self.x_prime_log_prior(x)
        else:
            log_p = self.log_prior(x)

        log_w = log_p - log_q
        if return_log_prior:
            return log_w, log_p
        else:
            return log_w

    def rejection_sampling(self, z, min_log_q=None):
        """
        Perform rejection sampling.

        Converts samples from the latent space and computes the corresponding
        weights. Then returns samples using standard rejection sampling.

        Parameters
        ----------
        z :  ndarray
            Samples from the latent space
        min_log_q : float, optional
            Lower bound on the log-probability computed using the flow that
            is used to truncate new samples.

        Returns
        -------
        array_like
            Array of accepted latent samples.
        array_like
            Array of accepted samples in the X space.
        """
        msg = (
            "`FlowProposal.rejection_sampling` is deprecated and will be "
            "removed in a future release."
        )
        warn(msg, FutureWarning)

        x, log_q, z = self.backward_pass(
            z,
            rescale=not self.use_x_prime_prior,
            discard_nans=False,
            return_z=True,
        )

        if not x.size:
            return np.array([]), x

        if min_log_q:
            above = log_q >= min_log_q
            x = x[above]
            z = z[above]
            log_q = log_q[above]
        else:
            valid = np.isfinite(log_q)
            x, z, log_q = get_subset_arrays(valid, x, z, log_q)

        # rescale given priors used initially, need for priors
        log_w = self.compute_weights(x, log_q)
        log_u = np.log(np.random.rand(x.shape[0]))
        indices = np.where(log_w >= log_u)[0]

        return z[indices], x[indices]

    def convert_to_samples(self, x, plot=True):
        """
        Convert the array to samples ready to be used.

        This removes are auxiliary parameters, (e.g. auxiliary radial
        parameters) and ensures the prior is computed. These samples can
        be directly used in the nested sampler.

        Parameters
        ----------
        x : array_like
            Array of samples
        plot : bool, optional
            If true a 1d histogram for each parameter of the pool is plotted.
            This includes a comparison the live points used to train the
            current realisation of the flow.

        Returns
        -------
        array
            Structured array of samples.
        """
        if self.use_x_prime_prior:
            if self._plot_pool and plot:
                plot_1d_comparison(
                    self.training_data_prime,
                    x,
                    labels=["live points", "pool"],
                    filename=os.path.join(
                        self.output, f"pool_prime_{self.populated_count}.png"
                    ),
                )

            x, _ = self.inverse_rescale(x)
        x["logP"] = self.model.batch_evaluate_log_prior(x)
        return rfn.repack_fields(
            x[self.model.names + config.livepoints.non_sampling_parameters]
        )

    def prep_latent_prior(self):
        """Prepare the latent prior."""
        if self.latent_prior == "truncated_gaussian":
            self._populate_dist = NDimensionalTruncatedGaussian(
                self.dims,
                self.r,
                fuzz=self.fuzz,
            )
            self._draw_func = self._populate_dist.sample
        elif self.latent_prior == "flow":
            self._draw_func = lambda N: self.flow.sample_latent_distribution(N)
        else:
            self._draw_func = partial(
                self._draw_latent_prior,
                dims=self.dims,
                r=self.r,
                fuzz=self.fuzz,
            )

    def draw_latent_prior(self, n):
        """Draw n samples from the latent prior."""
        return self._draw_func(N=n)

    def populate(
        self, worst_point, N=10000, plot=True, r=None, max_samples=1_000_000
    ):
        """
        Populate a pool of latent points given the current worst point.

        Parameters
        ----------
        worst_point : structured_array
            The current worst point used to compute the radius of the contour
            in the latent space.
        N : int, optional (10000)
            The total number of points to populate in the pool
        plot : {True, False, 'all'}
            Enable or disable plots for during training. By default the plots
            are only one-dimensional histograms, `'all'` includes corner
            plots with samples, these are often a few MB in size so
            proceed with caution!
        """
        st = datetime.datetime.now()
        if not self.initialised:
            raise RuntimeError(
                "Proposal has not been initialised. "
                "Try calling `initialise()` first."
            )
        if r is not None:
            logger.debug(f"Using user inputs for radius {r}")
        elif self.fixed_radius:
            r = self.fixed_radius
        else:
            logger.debug(f"Populating with worst point: {worst_point}")
            if self.compute_radius_with_all:
                logger.debug("Using previous live points to compute radius")
                worst_point = self.training_data
            worst_z = self.forward_pass(
                worst_point, rescale=True, compute_radius=True
            )[0]
            r = self.radius(worst_z)
            if self.max_radius and r > self.max_radius:
                r = self.max_radius
            if self.min_radius and r < self.min_radius:
                r = self.min_radius

        if self.truncate_log_q:
            log_q_live_points = self.forward_pass(self.training_data)[1]
            min_log_q = log_q_live_points.min()
            logger.debug(f"Truncating with log_q={min_log_q:.3f}")
        else:
            min_log_q = None

        logger.debug(f"Populating proposal with latent radius: {r:.5}")
        self.r = r

        self.alt_dist = self.get_alt_distribution()

        if self.indices:
            logger.debug(
                "Existing pool of samples is not empty. "
                "Discarding existing samples."
            )
        self.indices = []

        if self.accumulate_weights:
            samples = empty_structured_array(0, dtype=self.population_dtype)
        else:
            samples = empty_structured_array(N, dtype=self.population_dtype)

        self.prep_latent_prior()

        log_n = np.log(N)
        log_n_expected = -np.inf
        n_proposed = 0
        log_weights = np.empty(0)
        log_constant = -np.inf
        n_accepted = 0
        accept = None

        while n_accepted < N:
            z = self.draw_latent_prior(self.drawsize)
            n_proposed += z.shape[0]

            x, log_q = self.backward_pass(
                z, rescale=not self.use_x_prime_prior
            )
            if self.truncate_log_q:
                above_min_log_q = log_q > min_log_q
                logger.debug(
                    "Discarding %s samples below log_q_min",
                    self.drawsize - above_min_log_q.sum(),
                )
                x, log_q = get_subset_arrays(above_min_log_q, x, log_q)
            # Handle case where all samples are below min_log_q
            if not len(x):
                continue
            log_w = self.compute_weights(x, log_q)

            if self.accumulate_weights:

                samples = np.concatenate([samples, x])
                log_weights = np.concatenate([log_weights, log_w])
                log_constant = max(np.nanmax(log_w), log_constant)
                log_n_expected = logsumexp(log_weights - log_constant)

                logger.debug(
                    "Drawn %s - n expected: %s / %s",
                    samples.size,
                    np.exp(log_n_expected),
                    N,
                )

                # Only try rejection sampling if we expected to accept enough
                # points. In the case where we don't, we continue drawing
                # samples
                if log_n_expected >= log_n:
                    log_u = np.log(np.random.rand(len(log_weights)))
                    accept = (log_weights - log_constant) > log_u
                    n_accepted = np.sum(accept)
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break

            else:
                log_w -= log_w.max()
                log_u = np.log(np.random.rand(len(log_w)))
                accept = log_w > log_u
                n_accept_batch = accept.sum()
                m = min(N - n_accepted, n_accept_batch)
                samples[n_accepted : n_accepted + m] = x[accept][:m]
                n_accepted += n_accept_batch
                logger.debug("n accepted: %s / %s", n_accepted, N)

        if self.accumulate_weights:
            if accept is None or len(accept) != len(samples):
                log_u = np.log(np.random.rand(len(log_weights)))
                accept = (log_weights - log_constant) > log_u
            logger.debug("Total number of samples: %s", samples.size)
            n_accepted = np.sum(accept)
            self.x = samples[accept][:N]
        else:
            self.x = samples[:N]

        self.samples = self.convert_to_samples(self.x, plot=plot)

        if self._plot_pool and plot:
            self.plot_pool(self.samples)

        self.population_time += datetime.datetime.now() - st
        logger.debug("Evaluating log-likelihoods")
        self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
            self.samples
        )
        if self.check_acceptance:
            self.acceptance.append(
                self.compute_acceptance(worst_point["logL"])
            )
            logger.debug(f"Current acceptance {self.acceptance[-1]}")

        self.indices = np.random.permutation(self.samples.size).tolist()
        self.population_acceptance = n_accepted / n_proposed
        self.populated_count += 1
        self.populated = True
        self._checked_population = False
        logger.debug(f"Proposal populated with {len(self.indices)} samples")
        logger.debug(
            f"Overall proposal acceptance: {self.x.size / n_proposed:.4}"
        )

    def get_alt_distribution(self):
        """
        Get a distribution for the latent prior used to draw samples.
        """
        if self.latent_prior in ["uniform_nsphere", "uniform_nball"]:
            return get_uniform_distribution(
                self.dims, self.r * self.fuzz, device=self.flow.device
            )

    def compute_acceptance(self, logL):
        """
        Compute how many of the current pool have log-likelihoods greater
        than the specified log-likelihood using the current value in the
        `logL` field.

        Parameters
        ----------
        float : logL
            Log-likelihood to use as the lower bound

        Returns
        -------
        float :
            Acceptance defined on [0, 1]
        """
        return (self.samples["logL"] > logL).sum() / self.samples.size

    def draw(self, worst_point):
        """
        Draw a replacement point. The new point is independent of the worst
        point. The worst point is only used during population.

        Parameters
        ----------
        worst_point : structured_array
            The current worst point used to compute the radius of the contour
            in the latent space.

        Returns
        -------
        structured_array
            New live point
        """
        if not self.populated:
            self.populating = True
            if self.update_poolsize:
                self.update_poolsize_scale(self.ns_acceptance)
            while not self.populated:
                self.populate(worst_point, N=self.poolsize)
            self.populating = False
        # new sample is drawn randomly from proposed points
        # popping from right end is faster
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
            logger.debug("Proposal pool is empty")
        # make live point and return
        return new_sample

    @nessai_style()
    def plot_pool(self, x):
        """
        Plot the pool of points.

        Parameters
        ----------
        x : array_like
            Corresponding samples to plot in the physical space.
        """
        if self._plot_pool == "all":
            plot_live_points(
                x,
                c="logL",
                filename=os.path.join(
                    self.output, f"pool_{self.populated_count}.png"
                ),
            )
        else:
            plot_1d_comparison(
                self.training_data,
                x,
                labels=["live points", "pool"],
                filename=os.path.join(
                    self.output, f"pool_{self.populated_count}.png"
                ),
            )

            z, log_q = self.forward_pass(x, compute_radius=False)
            z_tensor = (
                torch.from_numpy(z)
                .type(torch.get_default_dtype())
                .to(self.flow.device)
            )
            with torch.inference_mode():
                if self.alt_dist is not None:
                    log_p = self.alt_dist.log_prob(z_tensor).cpu().numpy()
                else:
                    log_p = (
                        self.flow.model.base_distribution_log_prob(z_tensor)
                        .cpu()
                        .numpy()
                    )

            fig, axs = plt.subplots(3, 1, figsize=(3, 9))
            axs = axs.ravel()
            axs[0].hist(log_q, 20, histtype="step", label="log q")
            axs[1].hist(log_q - log_p, 20, histtype="step", label="log J")
            axs[2].hist(
                np.sqrt(np.sum(z**2, axis=1)),
                20,
                histtype="step",
                label="Latent radius",
            )
            axs[0].set_xlabel("Log q")
            axs[1].set_xlabel("Log |J|")
            axs[2].set_xlabel("r")
            plt.tight_layout()
            fig.savefig(
                os.path.join(
                    self.output, f"pool_{self.populated_count}_log_q.png"
                )
            )
            plt.close(fig)

    def resume(self, model, flow_config, weights_file=None):
        """
        Resume the proposal.

        The model and config are not stored so these must be provided.

        Parameters
        ----------
        model : :obj:`nessai.model.Model`
            User-defined model used.
        flow_config : dict
            Configuration dictionary for the flow.
        weights_files : str, optional
            Weights file to try and load. If not provided the proposal
            tries to load the last weights file.
        """
        super().resume(model)
        self.flow_config = flow_config
        self._reparameterisation = None

        if self.mask is not None:
            if isinstance(self.mask, list):
                m = np.array(self.mask)
            self.flow_config["model_config"]["kwargs"]["mask"] = m

        self.initialise()

        if weights_file is None:
            weights_file = self.weights_file

        # Flow might have exited before any weights were saved.
        if weights_file is not None:
            if os.path.exists(weights_file):
                self.flow.reload_weights(weights_file)
        else:
            logger.warning("Could not reload weights for flow")

        if self.update_bounds:
            if self.training_data is not None:
                self.check_state(self.training_data)
            elif self.training_data is None and self.training_count:
                raise RuntimeError("Could not resume! Missing training data!")

    def reset(self):
        """Reset the proposal"""
        self.indices = []
        self.samples = None
        self.x = None
        self.populated = False
        self.populated_count = 0
        self.population_acceptance = None
        self._poolsize_scale = 1.0
        self.r = np.nan
        self.alt_dist = None
        self._checked_population = True
        self.acceptance = []
        self._draw_func = None
        self._populate_dist = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["initialised"] = False
        state["weights_file"] = getattr(
            state.get("flow"), "weights_file", None
        )

        # Mask may be generate via permutation, so must be saved
        if "mask" in getattr(state.get("flow"), "model_config", {}).get(
            "kwargs", []
        ):
            state["mask"] = state["flow"].model_config["kwargs"]["mask"]
        else:
            state["mask"] = None
        if state["populated"] and state["indices"]:
            state["resume_populated"] = True
        else:
            state["resume_populated"] = False

        state["_draw_func"] = None
        state["_populate_dist"] = None

        # user provides model and config for resume
        # flow can be reconstructed from resume
        del state["_reparameterisation"]
        del state["model"]
        del state["_flow_config"]
        del state["flow"]

        return state
