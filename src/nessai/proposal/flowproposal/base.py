"""Base proposal class that contains common methods."""

import copy
import logging
import os
import re
from abc import abstractmethod
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
import torch

from ... import config
from ...flowmodel import FlowModel
from ...livepoint import (
    empty_structured_array,
    get_dtype,
    live_points_to_array,
    numpy_array_to_live_points,
)
from ...plot import nessai_style, plot_1d_comparison, plot_live_points
from ...reparameterisations import (
    CombinedReparameterisation,
    get_reparameterisation,
)
from ...utils import (
    save_live_points,
)
from ..rejection import RejectionProposal

logger = logging.getLogger(__name__)


class BaseFlowProposal(RejectionProposal):
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
    poolsize : int, optional
        Size of the proposal pool. Defaults to 10000.
    update_poolsize : bool, optional
        If True the poolsize is updated using the current acceptance of the
        nested sampler.
    max_poolsize_scale : int, optional
        Maximum scale for increasing the poolsize. E.g. if this value is 10
        and the poolsize is 1000 the maximum number of points in the pool
        is 10,000.
    check_acceptance : bool, optional
        If True the acceptance is computed after populating the pool. This
        includes computing the likelihood for every point. Default False.
    reparameterisations : Union[dict, str], optional
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
    map_to_unit_hypercube : bool
        If true, samples will be mapped to the unit hypercube before any
        reparameterisations are applied.
    """

    use_default_reparameterisations = False
    """
    Indicates whether reparameterisations will be used be default in this
    class. Child classes can change this value a force the default
    behaviour to change without changing the keyword arguments.
    """
    _FlowModelClass = FlowModel
    alt_dist = None

    def __init__(
        self,
        model,
        rng: Optional[np.random.Generator] = None,
        flow_config=None,
        training_config=None,
        output=None,
        poolsize=None,
        plot="min",
        check_acceptance=False,
        max_poolsize_scale=10,
        update_poolsize=True,
        accumulate_weights=False,
        save_training_data=False,
        reparameterisations=None,
        fallback_reparameterisation="zscore",
        use_default_reparameterisations=None,
        reverse_reparameterisations=False,
        map_to_unit_hypercube=False,
    ):
        super().__init__(model, rng=rng)

        self._x_dtype = None
        self._x_prime_dtype = None
        self._draw_func = None
        self._prior_bounds = None

        self.flow = None
        self._flow_config = None
        self._training_config = None
        self.populated = False
        self.populating = False  # Flag used for resuming during population
        self.indices = []
        self.training_count = 0
        self.populated_count = 0
        self.parameters = None
        self.training_data = None
        self.save_training_data = save_training_data
        self.x = None
        self.samples = None
        self.prime_parameters = None
        self.acceptance = []
        self._reparameterisation = None
        self.rescaling_set = False
        self.use_x_prime_prior = False
        self.should_update_reparameterisations = False
        self.map_to_unit_hypercube = map_to_unit_hypercube
        self.accumulate_weights = accumulate_weights

        self.reparameterisations = reparameterisations
        if use_default_reparameterisations is not None:
            self.use_default_reparameterisations = (
                use_default_reparameterisations
            )
        self.fallback_reparameterisation = fallback_reparameterisation
        self.reverse_reparameterisations = reverse_reparameterisations

        self.output = output if output is not None else os.getcwd()

        self.check_acceptance = check_acceptance
        self.flow_config = flow_config
        self.training_config = training_config

        self.configure_poolsize(
            poolsize=poolsize,
            max_poolsize_scale=max_poolsize_scale,
            update_poolsize=update_poolsize,
        )
        self.configure_plotting(plot)

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
        self._flow_config = config

    @property
    def training_config(self):
        """Return the configuration for the flow"""
        return self._training_config

    @training_config.setter
    def training_config(self, config):
        """Set training configuration."""
        if config is None:
            config = {}
        self._training_config = config

    @property
    def dims(self):
        """Return the number of dimensions"""
        return len(self.parameters)

    @property
    def rescaled_dims(self):
        """Return the number of rescaled dimensions"""
        return len(self.prime_parameters)

    @property
    def x_dtype(self):
        """Return the dtype for the x space"""
        if self._x_dtype is None:
            self._x_dtype = get_dtype(
                self.parameters, config.livepoints.default_float_dtype
            )
        return self._x_dtype

    @property
    def x_prime_dtype(self):
        """Return the dtype for the x prime space"""
        if self._x_prime_dtype is None:
            self._x_prime_dtype = get_dtype(
                self.prime_parameters, config.livepoints.default_float_dtype
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

    @property
    def prior_bounds(self):
        """The priors bounds used when computing the priors.

        If :code:`map_to_unit_hypercube` is true, these will be [0, 1]
        """
        if self._prior_bounds is None:
            if self.map_to_unit_hypercube:
                logger.debug("Setting prior bounds to the unit-hypercube")
                self._prior_bounds = {
                    n: np.array([0.0, 1.0]) for n in self.model.names
                }
            else:
                logger.debug("Setting prior bounds to the model prior bounds")
                self._prior_bounds = self.model.bounds
        return self._prior_bounds

    def configure_poolsize(
        self,
        poolsize,
        update_poolsize,
        max_poolsize_scale,
    ):
        """
        Configure settings related to the pool size
        """
        if poolsize is None:
            raise RuntimeError("Must specify `poolsize`")

        self._poolsize = poolsize
        self._poolsize_scale = 1.0
        self.update_poolsize = update_poolsize
        self.max_poolsize_scale = max_poolsize_scale
        self.ns_acceptance = 1.0

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

    def update_flow_config(self):
        """Update the flow configuration dictionary."""
        self.flow_config["n_inputs"] = self.rescaled_dims

    def initialise(self, resumed: bool = False) -> None:
        """
        Initialise the proposal class.

        This includes:
            * Setting up the rescaling
            * Verifying the rescaling is invertible
            * Initialising the FlowModel

        Parameters
        ----------
        resumed : bool
            Indicates if the proposal is being initialised after being resumed
            or not. When true, the reparameterisations will not be
            reinitialised.
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        # Initialise if not resuming or resuming but initialised is False
        if not resumed or not self.initialised:
            self.set_rescaling()
            self.verify_rescaling()

        self.update_flow_config()
        self.flow = self._FlowModelClass(
            flow_config=self.flow_config,
            training_config=self.training_config,
            output=self.output,
            rng=self.rng,
        )
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
        reparameterisations : Union[dict, str, None]
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
        if isinstance(_reparameterisations, str):
            _reparameterisations = {
                _reparameterisations: {"parameters": self.model.names}
            }
        elif not isinstance(_reparameterisations, dict):
            raise TypeError(
                "Reparameterisations must be a dictionary, string or None, "
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
                    p: self.prior_bounds[p]
                    for p in default_config["parameters"]
                }
            else:
                prior_bounds = {
                    default_config["parameters"]: self.prior_bounds[
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
                p: self.prior_bounds[p] for p in other_params
            }
            logger.info(
                f"Assuming fallback reparameterisation "
                f"({FallbackClass.__name__}) for {other_params} with kwargs: "
                f"{fallback_kwargs}."
            )
            r = FallbackClass(parameters=other_params, **fallback_kwargs)
            self._reparameterisation.add_reparameterisations(r)

        if any(r._update for r in self._reparameterisation.values()):
            self.should_update_reparameterisations = True
        else:
            self.should_update_reparameterisations = False

        if self._reparameterisation.has_prime_prior:
            self.use_x_prime_prior = True
            self.x_prime_log_prior = self._reparameterisation.x_prime_log_prior
            logger.debug("Using x prime prior")
            if self.map_to_unit_hypercube:
                raise RuntimeError(
                    "x prime prior does not support map to unit hypercube"
                )
        else:
            logger.debug("Prime prior is disabled")
            if self._reparameterisation.requires_prime_prior:
                raise RuntimeError(
                    "One or more reparameterisations require use of the x "
                    "prime prior but it cannot be enabled with the current "
                    "settings."
                )

        self._reparameterisation.check_order()

        self.parameters = self._reparameterisation.parameters
        self.prime_parameters = self._reparameterisation.prime_parameters

    @property
    def names(self):
        warn(
            "`names` is deprecated, use `parameters` instead",
            FutureWarning,
        )
        return self.parameters

    @property
    def rescaled_names(self):
        warn(
            (
                "`rescaled_names` is deprecated, use `prime_parameters` "
                "instead"
            ),
            FutureWarning,
        )
        return self.prime_parameters

    @property
    def update_bounds(self):
        warn(
            (
                "`update_bounds` is deprecated, use "
                "`should_update_reparameterisations` instead."
            ),
            FutureWarning,
        )
        return self.should_update_reparameterisations

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        if self.model.reparameterisations is not None:
            self.reparameterisations = self.model.reparameterisations

        self.configure_reparameterisations(self.reparameterisations)

        logger.info(f"x space parameters: {self.parameters}")
        logger.info(f"x prime space parameters: {self.prime_parameters}")
        self.rescaling_set = True

    def verify_rescaling(self):
        """
        Verify the rescaling functions are invertible
        """
        if not self.rescaling_set:
            raise RuntimeError(
                "Rescaling must be set before it can be verified"
            )
        if not self._reparameterisation.one_to_one:
            logger.warning(
                "Could not check if reparameterisation is invertible"
            )
            return
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
        self._reparameterisation.reset()

    def rescale(self, x, **kwargs):
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

        if self.map_to_unit_hypercube:
            x = self.model.to_unit_hypercube(x)

        x, x_prime, log_J = self._reparameterisation.reparameterise(
            x, x_prime, log_J, **kwargs
        )

        for p in config.livepoints.non_sampling_parameters:
            x_prime[p] = x[p]
        return x_prime, log_J

    def inverse_rescale(self, x_prime, return_unit_hypercube=False, **kwargs):
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

        if self.map_to_unit_hypercube and not return_unit_hypercube:
            x = self.model.from_unit_hypercube(x)

        return x, log_J

    def check_state(self, x):
        """Update the state of the proposal given some training data.

        Includes updating the reparameterisations.

        Parameters
        ----------
        x: array_like
            Array of training live points which can be used to set parameters
        """
        if self.map_to_unit_hypercube:
            x = self.model.to_unit_hypercube(x)
        self._reparameterisation.update(x)

    @nessai_style()
    def _plot_training_data(self, output):
        """Plot the training data and compare to the results"""
        z_training_data, _ = self.forward_pass(
            self.training_data, rescale=True
        )
        z_gen = self.flow.sample_latent_distribution(self.training_data.size)

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
            parameters=self.prime_parameters,
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
            raise RuntimeError(f"{self.__name__} is not initialised.")

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
            x_prime, self.prime_parameters, copy=True
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

    def forward_pass(self, x, rescale=True, **kwargs):
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
            x, log_J_rescale = self.rescale(x, **kwargs)
            log_J += log_J_rescale

        x = live_points_to_array(x, names=self.prime_parameters, copy=True)

        if x.ndim == 1:
            x = x[np.newaxis, :]
        z, log_prob = self.flow.forward_and_log_prob(x)

        return z, log_prob + log_J

    def backward_pass(self, z, rescale=True, **kwargs):
        """
        A backwards pass from the model (latent -> real)

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space
        rescale : bool, optional (True)
            Apply inverse rescaling function

        Returns
        -------
        x : array_like
            Samples in the data space
        log_j : array_like
            Log Jacobian determinant
        """
        # Compute the log probability
        x, log_j = self.flow.inverse(z)

        x = numpy_array_to_live_points(
            x.astype(config.livepoints.default_float_dtype),
            self.prime_parameters,
        )
        # Apply rescaling in rescale=True
        if rescale:
            x, log_j_rescale = self.inverse_rescale(x, **kwargs)
            # Include Jacobian for the rescaling
            log_j += log_j_rescale
        return x, log_j

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

    def unit_hypercube_log_prior(self, x):
        """
        Compute the prior in the unit hypercube space.

        Parameters
        ----------
        x : array
            Samples in the unit hypercube.
        """
        if self._reparameterisation:
            return self.model.batch_evaluate_log_prior_unit_hypercube(
                x
            ) + self._reparameterisation.log_prior(x)
        else:
            return self.model.batch_evaluate_log_prior_unit_hypercube(x)

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
        elif self.map_to_unit_hypercube:
            log_p = self.unit_hypercube_log_prior(x)
        else:
            log_p = self.log_prior(x)

        log_w = log_p - log_q
        if return_log_prior:
            return log_w, log_p
        else:
            return log_w

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
        elif self.map_to_unit_hypercube:
            x = self.model.from_unit_hypercube(x)
        x = rfn.repack_fields(
            x[self.model.names + config.livepoints.non_sampling_parameters]
        )
        x["logP"] = self.model.batch_evaluate_log_prior(x)
        return x

    @abstractmethod
    def populate(self, worst_point, n_samples=10000):
        raise NotImplementedError

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
                self.populate(worst_point, n_samples=self.poolsize)
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

        if self.mask is not None:
            if isinstance(self.mask, list):
                m = np.array(self.mask)
            self.flow_config["mask"] = m

        self.initialise(resumed=True)

        if weights_file is None:
            weights_file = self.weights_file

        # Flow might have exited before any weights were saved.
        if weights_file is not None:
            if os.path.exists(weights_file):
                self.flow.reload_weights(weights_file)
        else:
            logger.warning("Could not reload weights for flow")

    def reset(self):
        """Reset the proposal"""
        self.indices = []
        self.samples = None
        self.x = None
        self.populated = False
        self.populated_count = 0
        self.population_acceptance = None
        self._poolsize_scale = 1.0
        self._checked_population = True
        self.acceptance = []
        self._reparameterisation.reset()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["initialised"] = False
        state["weights_file"] = getattr(
            state.get("flow"), "weights_file", None
        )

        # Mask may be generate via permutation, so must be saved
        if "mask" in getattr(state.get("flow"), "flow_config", {}):
            state["mask"] = state["flow"].flow_config["mask"]
        else:
            state["mask"] = None
        if state["populated"] and state["indices"]:
            state["resume_populated"] = True
        else:
            state["resume_populated"] = False

        # user provides model and config for resume
        # flow can be reconstructed from resume
        del state["model"]
        del state["_flow_config"]
        del state["flow"]

        return state
