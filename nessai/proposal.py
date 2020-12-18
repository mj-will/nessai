import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import logsumexp
import torch

from .flowmodel import FlowModel, update_config
from .livepoint import (
        live_points_to_array,
        numpy_array_to_live_points,
        get_dtype,
        DEFAULT_FLOAT_DTYPE
        )
from .plot import plot_live_points, plot_acceptance, plot_1d_comparison
from .utils import (
    get_uniform_distribution,
    get_multivariate_normal,
    detect_edge,
    save_live_points,
    InterpolatedDistribution
    )

logger = logging.getLogger(__name__)


def _initialize_global_variables(model):
    """
    Store a global copy of the model for multiprocessing.
    """
    global _model
    _model = model


def _log_likelihood_wrapper(x):
    """
    Wrapper for the log likelihood
    """
    return _model.evaluate_log_likelihood(x)


class Proposal:
    """
    Base proposal object

    Parameters
    ----------
    model: obj
        User-defined model
    """
    def __init__(self, model):
        self.model = model
        self.populated = True
        self.initialised = False
        self.training_count = 0
        self.population_acceptance = None
        self.r = None
        self.pool = None

    def initialise(self):
        """
        Initialise the proposal
        """
        self.initialised = True

    def draw(self, old_param):
        """
        New a new point given the old point
        """
        return None

    def train(self, x, **kwargs):
        """
        Train the proposal method

        Parameters
        ----------
        x: array_like
            Array of live points to use for training
        kwargs:
            Any of keyword arguments
        """
        logger.info('This proposal method cannot be trained')

    def resume(self, model):
        """
        Resume the proposal with the model
        """
        self.model = model

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state


class AnalyticProposal(Proposal):
    """"
    Class for drawining from analytic priors

    This assumes the `new_point` method of the model draws points
    from the prior
    """
    def draw(self, old_param):
        """
        Draw directly from the analytic priors.

        Ouput is independent of the input
        """
        return self.model.new_point()


class RejectionProposal(Proposal):
    """
    Object for rejection sampling from the priors. Relies on the method
    `new_point` included in `flowproposal.model.Model`.

    Parameters
    ----------
    model : obj
        User-defined model
    poolsize : int
        Number of new samples to store in the pool
    """
    def __init__(self, model, poolsize=1000):
        super(RejectionProposal, self).__init__(model)
        self._poolsize = poolsize
        self.populated = False
        self._acceptance_checked = True

    @property
    def poolsize(self):
        return self._poolsize

    def draw_proposal(self):
        """Draw a signal new point"""
        return self.model.new_point(N=self.poolsize)

    def log_proposal(self, x):
        """
        Log proposal probability. Calls `model.new_point_log_prob`

        Parameters
        ----------
        x : structured_array
            Array of new points
        """
        return self.model.new_point_log_prob(x)

    def compute_weights(self, x):
        """
        Get weights for the samples.

        Computes the log weights for rejection sampling sampling such that
        that the maximum log probability is zero.

        Parameters
        ----------
        x :  structed_arrays
            Array of points
        """
        x['logP'] = self.model.log_prior(x)
        log_q = self.log_proposal(x)
        log_w = x['logP'] - log_q
        log_w -= np.nanmax(log_w)
        return log_w

    @property
    def population_acceptance(self):
        """
        Check the acceptance of the current proposal method.

        If this method has already been called since the proposal was
        last populated it returns None.
        """
        if self._acceptance_checked:
            return None
        else:
            self._acceptance_checked = True
            return self._population_acceptance

    @population_acceptance.setter
    def population_acceptance(self, acceptance):
        """
        Set the population acceptance and reset the flag
        """
        self._population_acceptance = acceptance
        self._acceptance_checked = False

    def populate(self):
        """
        Populate the pool by drawing from the proposal distribution and
        using rejection sampling.
        """
        x = self.draw_proposal()
        log_w = self.compute_weights(x)
        log_u = np.log(np.random.rand(self.poolsize))
        indices = np.where((log_w - log_u) >= 0)[0]
        self.samples = x[indices]
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        self.population_acceptance = self.samples.size / self.poolsize
        self.populated = True

    def draw(self, old_sample):
        """
        Propose a new sample. Draws from the pool if it is populated, else
        it populates the pool.

        Parameters
        ----------
        old_sample : structured_array
            Old sample, this is not used in the proposal method
        """
        if not self.populated:
            self.populate()
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
        return new_sample


class FlowProposal(RejectionProposal):
    """
    Object that handles training and proposal points

    Parameters
    ----------
    model: obj:`flowproposal.model.Model`
        User defined model
    flow_config: dict, optional
        Configuration for training the normalising flow. If None, uses default
        settings. Defaults to None.
    output: str, optional
        Path to output directory. Defaults to ./
    plot: {True, False, 'all', 'min'}, optional
        Controls the plotting level:
        * True: all plots
        * False: no plots
        * 'all': all plots
        * 'min': 1d plots and loss
        Defaults to `'min'`
    latent_prior: {'truncated_gaussian', 'gaussian', 'uniform_nsphere',
                   'gaussian'}, optional
        Prior distribution in the latent space. Defaults to
        'truncated_gaussian'.
    poolsize: int, optional
        Size of the proposal pool. Defaults to 10000.
    drawsize: int, optional
        Number of points to simultaneosly draw when populating the proposal
        Defaults to 10000
    check_acceptance: bool, optional
        If True the acceptance is computed after populating the pool. This
        includes computing the likelihood for every point. Default False.
    max_radius: float (optional)
        If a float then this value is used as an upper limit for the
        computed radius when populating the proposal. If unspecified no
        upper limit is used.
    fuzz: float (optional)
        Fuzz-factor applied to the radius. If unspecified no fuzz-factor is
        applied.
    zero_reset: int (optional)
        Value used when reseting proposal if zero samples are accepted.
        If specified is after drawing samples zero_reset times the current
        poolsize is less than half, then the radius is reduced by 1% and
        the pool is reset.
    truncate: bool (optional)
        Truncate proposals using probability compute for worst point.
        Not recommended.
    rescale_parameters: list or bool (optional)
        If True live points are rescaled to `rescale_bounds` before training.
        If an instance of `list` then must contain names of parameters to
        rescale. If False no rescaling is applied. Default False.
    rescale_bounds: list (optional)
        Lower and upper bound to use for rescaling. Defaults to [-1, 1]. See
        `rescale_parameters`.
    update_bounds: bool (optional)
        If True bounds used for rescaling are updated at the starting of
        training. If False prior bounds are used. Default False.
    boundary_inversion: bool or list (optional)
        If True boundary inversion is applied to all bounds. If
        If an instance of `list` of parameters names, then inversion
        only applied to these parameters. If False (default )no inversion is
        used.
    """

    def __init__(self, model, flow_config=None, output='./', poolsize=10000,
                 rescale_parameters=False, latent_prior='truncated_gaussian',
                 fuzz=1.0, keep_samples=False, plot='min',
                 fixed_radius=False, drawsize=10000, check_acceptance=False,
                 truncate=False, zero_reset=None,
                 rescale_bounds=[-1, 1], expansion_fraction=0.0,
                 boundary_inversion=False, inversion_type='duplicate',
                 update_bounds=True, max_radius=False, pool=None, n_pool=None,
                 multiprocessing=False, max_poolsize_scale=50,
                 update_poolsize=False, save_training_data=False,
                 compute_radius_with_all=False, draw_latent_kwargs={},
                 detect_edges=False, detect_edges_kwargs={},
                 **kwargs):
        """
        Initialise
        """
        super(FlowProposal, self).__init__(model)
        logger.debug('Initialising FlowProposal')

        self.flow = None
        self._flow_config = None
        self.initialised = False
        self.populated = False
        self.populating = False    # Flag used for resuming during population
        self.indices = []
        self.training_count = 0
        self.populated_count = 0
        self.population_time = datetime.timedelta()
        self.logl_eval_time = datetime.timedelta()
        self.names = []
        self.training_data = None
        self.save_training_data = save_training_data
        self.x = None
        self.samples = None
        self.rescaled_names = []
        self.acceptance = []
        self.approx_acceptance = []
        self._edges = {}
        self._inversion_test_type = None
        self.use_x_prime_prior = False

        self.output = output
        self._poolsize = poolsize
        self._poolsize_scale = 1.0
        self.update_poolsize = update_poolsize
        self.max_poolsize_scale = max_poolsize_scale
        self.ns_acceptance = 1.
        self.drawsize = drawsize
        self.fuzz = fuzz
        self.expansion_fraction = expansion_fraction
        self.latent_prior = latent_prior
        self.rescale_parameters = rescale_parameters
        self.keep_samples = keep_samples
        self.update_bounds = update_bounds
        self.check_acceptance = check_acceptance
        self.rescale_bounds = rescale_bounds
        self.truncate = truncate
        self.zero_reset = zero_reset
        self.boundary_inversion = boundary_inversion
        self.inversion_type = inversion_type
        self.flow_config = flow_config

        self.detect_edges = detect_edges
        self.configure_edge_detection(detect_edges_kwargs)

        self.compute_radius_with_all = compute_radius_with_all
        self.max_radius = float(max_radius)
        self.configure_fixed_radius(fixed_radius)

        self.pool = pool
        self.n_pool = n_pool
        if multiprocessing:
            logger.info('Using multiprocessing')
            if not self.check_acceptance:
                self.check_acceptance = True
            self.setup_pool()

        self.configure_plotting(plot)

        self.clip = self.flow_config.get('clip', False)

        self.draw_latent_kwargs = draw_latent_kwargs
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
        """Set configuration (includes checking defaults)"""
        self._flow_config = update_config(config)

    @property
    def dims(self):
        """Return the number of dimensions"""
        return len(self.names)

    @property
    def rescaled_dims(self):
        """Return the number of rescaled dimensions"""
        return len(self.rescaled_names)

    @property
    def flow_dims(self):
        """Return the number of dimensions for the normalising flow"""
        return self.rescaled_dims

    @property
    def x_dtype(self):
        """Return the dtype for the x space"""
        return get_dtype(self.names, DEFAULT_FLOAT_DTYPE)

    @property
    def x_prime_dtype(self):
        """Return the dtype for the x prime space"""
        return get_dtype(self.rescaled_names, DEFAULT_FLOAT_DTYPE)

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

    def setup_pool(self):
        """
        Setup the multiprocessing pool
        """
        if self.pool is None:
            logger.info(
                f'Starting multiprocessing pool with {self.n_pool} processes')
            import multiprocessing
            self.pool = multiprocessing.Pool(
                processes=self.n_pool,
                initializer=_initialize_global_variables,
                initargs=(self.model,)
            )

    def close_pool(self):
        """
        Close the the multiprocessing pool
        """
        if getattr(self, "pool", None) is not None:
            logger.info("Starting to close worker pool.")
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Finished closing worker pool.")

    def configure_plotting(self, plot):
        """Configure plotting"""
        if plot:
            if isinstance(plot, str):
                if plot == 'all':
                    self._plot_pool = 'all'
                    self._plot_training = 'all'
                elif plot == 'train':
                    self._plot_pool = False
                    self._plot_training = 'all'
                elif plot == 'pool':
                    self._plot_pool = True
                    self._plot_training = False
                elif plot == 'minimal' or plot == 'min':
                    self._plot_pool = True
                    self._plot_training = True
                else:
                    logger.warning(
                        f'Unknown plot argument: {plot}, setting all false'
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
        if self.latent_prior == 'truncated_gaussian':
            from .utils import draw_truncated_gaussian
            self.draw_latent_prior = draw_truncated_gaussian
            if k := (self.flow_config['model_config'].get('kwargs', {})):
                if v := (k.get('var', False)):
                    if 'var' not in self.draw_latent_kwargs:
                        self.draw_latent_kwargs['var'] = v

        elif self.latent_prior == 'gaussian':
            logger.warning('Using a gaussian latent prior WITHOUT truncation')
            from .utils import draw_gaussian
            self.draw_latent_prior = draw_gaussian
        elif self.latent_prior == 'uniform':
            from .utils import draw_uniform
            self.draw_latent_prior = draw_uniform
        elif self.latent_prior in ['uniform_nsphere', 'uniform_nball']:
            from .utils import draw_nsphere
            self.draw_latent_prior = draw_nsphere
        else:
            raise RuntimeError(
                f'Unknown latent prior: {self.latent_prior}, choose from: '
                'truncated_gaussian (default), gaussian, '
                'uniform, uniform_nsphere'
                )

    def configure_fixed_radius(self, fixed_radius):
        """Configure the fixed radius"""
        if fixed_radius:
            try:
                self.fixed_radius = float(fixed_radius)
            except ValueError:
                logger.error(
                    'Fixed radius enabled but could not be converted to a '
                    'float. Setting fixed_radius=False'
                    )
                self.fixed_radius = False
        else:
            self.fixed_radius = False

    def configure_edge_detection(self, d):
        """Configure parameters for edge detection"""
        default = dict(cutoff=0.5)
        if self.detect_edges:
            d['allow_none'] = True
        else:
            d['allow_none'] = False
            d['cutoff'] = 0.0
        default.update(d)
        self.detect_edges_kwargs = default
        logger.debug(f'detect edges kwargs: {self.detect_edges_kwargs}')

    def initialise(self):
        """
        Initialise the proposal class.

        This includes:
            * Setting up the rescaling
            * Verifying the rescaling is invertible
            * Intitialising the FlowModel
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.set_rescaling()
        self.verify_rescaling()
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info('Overwritting fuzz factor with expansion fraction')
            self.fuzz = \
                (1 + self.expansion_fraction) ** (1 / self.flow_dims)
            logger.info(f'New fuzz factor: {self.fuzz}')
        self.flow_config['model_config']['n_inputs'] = self.flow_dims
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
        logger.debug(f'Updating poolsize with acceptance: {acceptance:.3f}')
        if not acceptance:
            logger.warning('Acceptance is zero, using maximum scale')
            self._poolsize_scale = self.max_poolize_scale
        else:
            self._poolsize_scale = 1.0 / acceptance
            if self._poolsize_scale > self.max_poolsize_scale:
                logger.warning(
                    'Poolsize scaling is greater than maximum value')
                self._poolsize_scale = self.max_poolsize_scale
            if self._poolsize_scale < 1.:
                self._poolsize_scale = 1.

    def set_boundary_inversion(self):
        """
        Setup boundary inversion
        """
        if self.boundary_inversion:
            if not self.rescale_parameters:
                raise RuntimeError('Boundary inversion requires rescaling')

            if (isinstance(self.boundary_inversion, list) and
                    not set(self.boundary_inversion).issubset(self.names)):
                raise RuntimeError(
                            'Boundaries are not in known parameters')
            elif isinstance(self.rescale_parameters, list):
                self.boundary_inversion = self.rescale_parameters
            else:
                self.boundary_inversion = self.names.copy()

            logger.info('Appyling boundary inversion to: '
                        f'{self.boundary_inversion}')

            if self.inversion_type not in ('split', 'duplicate'):
                raise RuntimeError(
                        f'Unknown inversion type: {self.inversion_type}')

            self.rescale_bounds = [0, 1]
            self.update_bounds = True
            self._edges = {n: None for n in self.boundary_inversion}
            logger.info(f'Changing bounds to {self.rescale_bounds}')
        else:
            self.boundary_inversion = []

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()

        self.set_boundary_inversion()

        if self.rescale_parameters:
            # if rescale is a list, there are the parameters to rescale
            # else all parameters are rescale
            if not isinstance(self.rescale_parameters, list):
                self.rescale_parameters = self.names.copy()

            for i, rn in enumerate(self.rescaled_names):
                if rn in self.rescale_parameters:
                    self.rescaled_names[i] += '_prime'

            self._min = {n: self.model.bounds[n][0] for n in self.model.names}
            self._max = {n: self.model.bounds[n][1] for n in self.model.names}
            self._rescale_factor = np.ptp(self.rescale_bounds)
            self._rescale_shift = self.rescale_bounds[0]

            self.rescale = self._rescale_to_bounds
            self.inverse_rescale = self._inverse_rescale_to_bounds
            logger.info(f'Set to rescale inputs to {self.rescale_bounds}')

            if self.update_bounds:
                logger.info(
                        'Rescaling will use min and max of current live points'
                        )
            else:
                logger.info('Rescaling will use model bounds')

        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def verify_rescaling(self):
        """
        Verify the rescaling functions are invertible
        """
        logger.info('Verifying rescaling functions')
        x = self.model.new_point(N=1000)
        for inversion in ['lower', 'upper', 'both', False]:
            self.check_state(x)
            logger.debug(f'Testing: {inversion}')
            self._inversion_test_type = inversion
            x_prime, log_J = self.rescale(x)
            x_out, log_J_inv = self.inverse_rescale(x_prime)
            if x.size == x_out.size:
                for f in x.dtype.names:
                    if not np.allclose(x[f], x_out[f]):
                        raise RuntimeError(
                            f'Rescaling is not invertible for {f}')
                if not np.allclose(log_J, -log_J_inv):
                    raise RuntimeError('Rescaling Jacobian is not invertible')
            else:
                # ratio = x_out.size // x.size
                for f in x.dtype.names:
                    if not any([np.any(np.isclose(x[f], xo))
                                for xo in x_out[f]]):
                        raise RuntimeError(
                            'Duplicate samples must map to same input values. '
                            'Check the rescaling and inverse rescaling '
                            'functions.')
                for f in x.dtype.names:
                    if not np.allclose(x[f], x_out[f][:x.size]):
                        raise RuntimeError(
                            f'Rescaling is not invertible for {f}')
                if not np.allclose(log_J, -log_J_inv):
                    raise RuntimeError('Rescaling Jacobian is not invertible')

        self._inversion_test_type = None
        logger.info('Rescaling functions are invertible')

    def _rescale_to_bounds(self, x, compute_radius=False):
        """
        Rescale the inputs to specified bounds
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = np.zeros(x_prime.size)

        if x.size == 1:
            x = np.array([x], dtype=x.dtype)

        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x_prime[rn] = self._rescale_factor \
                             * ((x[n] - self._min[n])
                                / (self._max[n] - self._min[n])) \
                             + self._rescale_shift

                log_J += (-np.log(self._max[n] - self._min[n])
                          + np.log(self._rescale_factor))

                if n in self.boundary_inversion:

                    if self._edges[n] is None:
                        self._edges[n] = detect_edge(
                            x_prime[rn],
                            test=self._inversion_test_type,
                            **self.detect_edges_kwargs
                            )

                    if self._edges[n]:
                        logger.debug(
                            f'Apply inversion for {n} to '
                            f'{self._edges[n]} bound'
                            )
                        if self._edges[n] == 'upper':
                            x_prime[rn] = 1 - x_prime[rn]
                        if (self.inversion_type == 'duplicate' or
                                compute_radius):
                            x_inv = x_prime.copy()
                            x_inv[rn] *= -1
                            x_prime = np.concatenate([x_prime, x_inv])
                            x = np.concatenate([x,  x])
                            log_J = np.concatenate([log_J, log_J])
                        else:
                            inv = np.random.choice(x_prime.size,
                                                   x_prime.size // 2,
                                                   replace=False)
                            x_prime[rn][inv] *= -1
                    else:
                        logger.debug(f'Not using inversion for {n}')
            else:
                x_prime[rn] = x[n]
        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']
        return x_prime, log_J

    def _inverse_rescale_to_bounds(self, x_prime):
        """
        Rescale the inputs from the prime space to the phyiscal space
        using the bounds specified
        """
        x = np.zeros([x_prime.size], dtype=self.x_dtype)
        log_J = np.zeros(x_prime.size)
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                if n in self.boundary_inversion:
                    inv = x_prime[rn] < 0.
                    x_prime[rn][~inv] = x_prime[rn][~inv]
                    x_prime[rn][inv] = -x_prime[rn][inv]

                    if self._edges[n] == 'upper':
                        x_prime[rn] = 1 - x_prime[rn]

                x[n] = (self._max[n] - self._min[n]) \
                    * (x_prime[rn] - self._rescale_shift) \
                    / self._rescale_factor + self._min[n]

                log_J += (np.log(self._max[n] - self._min[n])
                          - np.log(self._rescale_factor))
            else:
                x[n] = x_prime[rn]
        x['logP'] = x_prime['logP']
        x['logL'] = x_prime['logL']
        return x, log_J

    def rescale(self, x, compute_radius=False):
        """
        Rescale from the phyisical space to the primed physical space

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
        log_J = np.zeros(x.size)
        return x, log_J

    def inverse_rescale(self, x_prime, **kwargs):
        """
        Rescale from the primed phyisical space to the original physical
        space

        Parameters
        ----------
        x_prime: array_like
            Array of live points to rescale

        Returns
        -------
        array
            Array of rescaled values in the data space
        array
            Array of log det|J|
        """
        log_J = np.zeros(x_prime.size)
        return x_prime, log_J

    def check_state(self, x):
        """
        Operations that need to checked before training. These include
        updating the bounds for rescaling and resetting the bounds for
        inversion.

        Parameters
        ----------
        x: array_like
            Array of training live points which can be used to set parameters
        """
        if self.update_bounds:
            self._min = {n: np.min(x[n]) for n in self.names}
            self._max = {n: np.max(x[n]) for n in self.names}
        if self.boundary_inversion:
            self._edges = {n: None for n in self.boundary_inversion}

    def train_on_data(self, x_prime, output):
        """
        Function that takes live points converts to numpy array and calls
        the train function. Live points should be in the X' (x prime) space.
        """
        x_prime_array = live_points_to_array(x_prime, self.rescaled_names)
        self.flow.train(x_prime_array,
                        output=output, plot=self._plot_training)

    def train(self, x, plot=True):
        """
        Train the normalising flow given some of the live points.

        Parameters
        ----------
        x : structured_array
            Array of live points
        plot : {True, False, 'all'}
            Enable or disable plots for during training. By default the plots
            are only one-dimenensional histograms, `'all'` includes corner
            plots with samples, these are often a fwe MB in size so
            proceed with caution!
        """
        if (plot and self._plot_training) or self.save_training_data:
            block_output = \
                self.output + f'/training/block_{self.training_count}/'
        else:
            block_output = self.output

        if not os.path.exists(block_output):
            os.makedirs(block_output, exist_ok=True)

        if self.save_training_data:
            save_live_points(x, f'{block_output}/training_data.json')

        self.training_data = x.copy()
        self.check_state(self.training_data)

        if self._plot_training == 'all' and plot:
            plot_live_points(x, c='logL',
                             filename=block_output + 'x_samples.png')

        x_prime, log_J = self.rescale(x)

        self.training_data_prime = x_prime.copy()

        if self.rescale_parameters and self._plot_training == 'all' and plot:
            plot_live_points(x_prime, c='logL',
                             filename=block_output + 'x_prime_samples.png')

        self.train_on_data(self.training_data_prime, block_output)

        if self._plot_training and plot:
            z_training_data, _ = self.forward_pass(self.training_data,
                                                   rescale=True)
            z_gen = np.random.randn(x.size, self.flow_dims)

            plot_1d_comparison(z_training_data, z_gen,
                               labels=['z_live_points', 'z_generated'],
                               convert_to_live_points=True,
                               filename=block_output + 'z_comparison.png')
            x_prime_gen, log_prob = self.backward_pass(z_gen, rescale=False)
            x_prime_gen['logL'] = log_prob
            x_gen, log_J = self.inverse_rescale(x_prime_gen)
            x_gen, log_J, x_prime_gen = \
                self.check_prior_bounds(x_gen, log_J, x_prime_gen)
            x_gen['logL'] += log_J
            if self._plot_training == 'all':
                plot_live_points(
                    x_prime_gen,
                    c='logL',
                    filename=block_output + 'x_prime_generated.png'
                    )
            if self.rescale_parameters:
                if self._plot_training == 'all':
                    plot_live_points(x_gen, c='logL',
                                     filename=block_output + 'x_generated.png')
                plot_1d_comparison(
                    x_prime, x_prime_gen, parameters=self.rescaled_names,
                    labels=['live points', 'generated'],
                    filename=block_output + 'x_prime_comparison.png')

            plot_1d_comparison(x, x_gen, parameters=self.model.names,
                               labels=['live points', 'generated'],
                               filename=block_output + 'x_comparison.png')

        self.populated = False
        self.training_count += 1

    def reset_model_weights(self, **kwargs):
        """
        Reset the flow weights
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
            Aditional arrays which correspond to the array of live points.
            Only those corresponding to points within the prior bounds
            are returned

        Returns
        -------
        out: tuple of arrays
            Array containing the subset of the orignal arrays which fall within
            the prior bounds
        """
        idx = np.array(list(((x[n] >= self.model.bounds[n][0])
                             & (x[n] <= self.model.bounds[n][1]))
                       for n in self.model.names)).T.all(1)
        out = (a[idx] for a in (x,) + args)
        return out

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
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            jacobian)
        """
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x, compute_radius=compute_radius)
            log_J += log_J_rescale

        x = live_points_to_array(x, names=self.rescaled_names)

        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[0] == 1:
            if self.clip:
                x = np.clip(x, *self.clip)
        z, log_prob = self.flow.forward_and_log_prob(x)

        return z, log_prob + log_J

    def backward_pass(self, z, rescale=True):
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
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            Jacobian)
        """
        # Compute the log probability
        try:
            x, log_prob = self.flow.sample_and_log_prob(
                z=z, alt_dist=self.alt_dist)
        except AssertionError:
            return np.array([]), np.array([])

        valid = np.isfinite(log_prob)
        x, log_prob = x[valid], log_prob[valid]
        x = numpy_array_to_live_points(x.astype(DEFAULT_FLOAT_DTYPE),
                                       self.rescaled_names)
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_prob -= log_J
            x, log_prob = self.check_prior_bounds(x, log_prob)
        return x, log_prob

    def radius(self, z, log_q=None):
        """
        Calculate the radius of a latent point or set of latent points.
        If multiple points are parsed the maximum radius is returned.

        Parameters
        ----------
        z : :obj:`np.ndarray`
            Array of points in the latent space
        log_q : :obj:`np.ndarray`, optional (None)
            Array of correponding probabilities. If specified
            then probability of the maximum radius is also returned.

        Returns
        -------
        tuple of arrays
            Tuple of array with the maximum raidus and correspoding log_q
            if it was a specified input.
        """
        if log_q is not None:
            r = np.sqrt(np.sum(z ** 2., axis=-1))
            i = np.argmax(r)
            return r[i], log_q[i]
        else:
            return np.nanmax(np.sqrt(np.sum(z ** 2., axis=-1)))

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
        return self.model.log_prior(x)

    def log_prior_x_prime(self, x):
        raise NotImplementedError()

    def compute_weights(self, x, log_q):
        """
        Compute weights for the samples.

        Computes the log weights for rejection sampling sampling such that
        that the maximum log probability is zero.

        Also sets the fields `logP` and `logL`. Note `logL` is set as the
        proposal probability.

        Parameters
        ----------
        x :  structed_arrays
            Array of points
        log_q : array_like
            Array of log proposal probabilties.
        """
        if self.use_x_prime_prior:
            x['logP'] = self.log_prior_x_prime(x)
        else:
            x['logP'] = self.log_prior(x)

        x['logL'] = log_q
        log_w = x['logP'] - log_q
        log_w -= np.max(log_w)
        return log_w

    def rejection_sampling(self, z, worst_q=None):
        """
        Perform rejection sampling
        """
        x, log_q = self.backward_pass(z, rescale=not self.use_x_prime_prior)

        if not x.size:
            return x, log_q

        if self.truncate:
            cut = log_q >= worst_q
            x = x[cut]
            log_q = log_q[cut]

        # rescale given priors used initially, need for priors
        log_w = self.compute_weights(x, log_q)
        log_u = np.log(np.random.rand(x.shape[0]))
        indices = np.where(log_w >= log_u)[0]

        return z[indices], x[indices]

    def convert_to_samples(self, x, plot=True):
        """
        Convert the array to samples ready to be used
        """
        if self.use_x_prime_prior:
            if self._plot_pool and plot:
                plot_1d_comparison(
                    self.training_data_prime, x,
                    labels=['live points', 'pool'],
                    filename=(f'{self.output}/pool_prime_'
                              + f'{self.populated_count}.png'))

            x, _ = self.inverse_rescale(x)
        return x[self.model.names + ['logP', 'logL']]

    def populate(self, worst_point, N=10000, plot=True):
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
            are only one-dimenensional histograms, `'all'` includes corner
            plots with samples, these are often a fwe MB in size so
            proceed with caution!
        """
        if self.fixed_radius:
            r = self.fixed_radius
        else:
            logger.debug(f'Populating with worst point: {worst_point}')
            if self.compute_radius_with_all:
                logger.debug('Using previous live points to compute radius')
                worst_point = self.training_data
            worst_z, worst_q = self.forward_pass(worst_point,
                                                 rescale=True,
                                                 compute_radius=True)
            r, worst_q = self.radius(worst_z, worst_q)
            if self.max_radius:
                if r > self.max_radius:
                    r = self.max_radius
            logger.info(f'Populating proposal with lantent radius: {r:.5}')
        self.r = r

        self.alt_dist = self.get_alt_distribution()

        if not self.keep_samples or not self.indices:
            self.x = np.empty(N,  dtype=self.population_dtype)
            self.indices = []
            z_samples = np.empty([N, self.flow_dims])

        proposed = 0
        accepted = 0
        percent = 0.1
        warn = True

        while accepted < N:

            z = self.draw_latent_prior(self.flow_dims, r=self.r,
                                       N=self.drawsize, fuzz=self.fuzz,
                                       **self.draw_latent_kwargs)

            proposed += z.shape[0]

            z, x = self.rejection_sampling(z, worst_q)

            if not x.size:
                continue

            if warn:
                if x.size / self.drawsize < 0.01:
                    logger.warning(
                        'Rejection sampling accepted less than 1 percent of '
                        f'samples! ({x.size / self.drawsize})')
                    warn = False

            n = min(x.size, N - accepted)
            self.x[accepted:(accepted+n)] = x[:n]
            z_samples[accepted:(accepted+n), ...] = z[:n]
            accepted += n
            if accepted > percent * N:
                logger.info(f'Accepted {accepted} / {N} points, '
                            f'acceptance: {accepted/proposed:.4}')
                percent += 0.1

        self.samples = self.convert_to_samples(self.x, plot=plot)

        if self._plot_pool and plot:
            self.plot_pool(z_samples, self.samples)

        if self.check_acceptance:
            self.approx_acceptance.append(self.compute_acceptance(worst_q))
            logger.debug(
                f'Current approximate acceptance {self.approx_acceptance[-1]}')
            st = datetime.datetime.now()
            self.evaluate_likelihoods()
            self.logl_eval_time += (datetime.datetime.now() - st)
            self.acceptance.append(
                self.compute_acceptance(worst_point['logL']))
            logger.debug(f'Current acceptance {self.acceptance[-1]}')
        else:
            self.samples['logL'] = np.zeros(self.samples.size,
                                            dtype=self.samples['logL'].dtype)

        self.indices = np.random.permutation(self.samples.size).tolist()
        self.population_acceptance = self.x.size / proposed
        self.populated_count += 1
        self.populated = True
        logger.info(f'Proposal populated with {len(self.indices)} samples')
        logger.info(
            f'Overall proposal acceptance: {self.x.size / proposed:.4}')

    def get_alt_distribution(self):
        """
        Get a distribution for the latent prior used to draw samples.
        """
        if self.latent_prior in ['uniform_nsphere', 'uniform_nball']:
            return get_uniform_distribution(self.flow_dims, self.r * self.fuzz,
                                            device=self.flow.device)
        elif self.latent_prior == 'truncated_gaussian':
            if 'var' in self.draw_latent_kwargs:
                return get_multivariate_normal(
                    self.flow_dims, var=self.draw_latent_kwargs['var'],
                    device=self.flow.device)

    def evaluate_likelihoods(self):
        """
        Evaluate the likelihoods for the pool of live points.

        If the multiprocessing pool has been started, the samples will be map
        using `pool.map`.
        """
        if self.pool is None:
            for s in self.samples:
                s['logL'] = self.model.evaluate_log_likelihood(s)
        else:
            self.samples['logL'] = self.pool.map(_log_likelihood_wrapper,
                                                 self.samples)
            self.model.likelihood_evaluations += self.samples.size

    def compute_acceptance(self, logL):
        """
        Compute how many of the current pool have log-likelihoods greater
        than the specified log-likelihood using the current value in the
        `logL` field.

        Parameters
        ----------
        float: logL
            Log-likelihood to use as the lower bound

        Returns
        -------
        float: acceptance
            Acceptance defined on [0, 1]
        """
        return (self.samples['logL'] > logL).sum() / self.samples.size

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
            st = datetime.datetime.now()
            while not self.populated:
                self.populate(worst_point, N=self.poolsize)
            self.population_time += (datetime.datetime.now() - st)
            self.populating = False
        # new sample is drawn randomly from proposed points
        # popping from right end is faster
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
            logger.debug('Proposal pool is empty')
        # make live point and return
        return new_sample

    def plot_pool(self, z, x):
        """
        Plot the pool
        """
        if self._plot_pool == 'all':
            plot_live_points(
                x, c='logL',
                filename=f'{self.output}/pool_{self.populated_count}.png')
        else:
            plot_1d_comparison(
                self.training_data, x, labels=['live points', 'pool'],
                filename=f'{self.output}/pool_{self.populated_count}.png')

            z_tensor = torch.from_numpy(z).to(self.flow.device)
            with torch.no_grad():
                if self.alt_dist is not None:
                    log_p = self.alt_dist.log_prob(z_tensor).cpu().numpy()
                else:
                    log_p = self.flow.model.base_distribution_log_prob(
                        z_tensor).cpu().numpy()

            fig, axs = plt.subplots(3, 1, figsize=(3, 9))
            axs = axs.ravel()
            axs[0].hist(x['logL'], 20, histtype='step', label='log q')
            axs[1].hist(x['logL'] - log_p, 20, histtype='step',
                        label='log J')
            axs[2].hist(np.sqrt(np.sum(z ** 2, axis=1)), 20,
                        histtype='step', label='Latent radius')
            axs[0].set_xlabel('Log q')
            axs[1].set_xlabel('Log |J|')
            axs[2].set_xlabel('r')
            plt.tight_layout()
            fig.savefig(
                f'{self.output}/pool_{self.populated_count}_log_q.png')

    def resume(self, model, flow_config, weights_file=None):
        """
        Resume the proposal
        """
        self.model = model
        self.flow_config = flow_config

        if (m := self.mask) is not None:
            if isinstance(m, list):
                m = np.array(m)
            self.flow_config['model_config']['kwargs']['mask'] = m

        self.initialise()

        if weights_file is None:
            weights_file = self.weights_file

        # Flow might have exited before any weights were saved.
        if weights_file is not None:
            if os.path.exists(weights_file):
                self.flow.reload_weights(weights_file)
        else:
            logger.warning('Could not reload weights for flow')

        if self.update_bounds:
            if self.training_data is not None:
                self.check_state(self.training_data)
            elif self.training_data is None and self.training_count:
                raise RuntimeError(
                    'Could not resume! Missing training data!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['initialised'] = False
        state['weights_file'] = state['flow'].weights_file
        # Mask may be generate via permutation, so must be saved
        if 'mask' in state['flow'].model_config['kwargs']:
            state['mask'] = state['flow'].model_config['kwargs']['mask']
        else:
            state['mask'] = None
        state['pool'] = None
        if state['populated'] and self.indices:
            state['resume_populated'] = True
        else:
            state['resumed_populated'] = False

        # user provides model and config for resume
        # flow can be reconstructed from resume
        del state['model']
        del state['_flow_config']
        del state['flow']
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class ConditionalFlowProposal(FlowProposal):

    def __init__(self, model, uniform_parameters=False,
                 conditional_likelihood=False, **kwargs):
        super(ConditionalFlowProposal, self).__init__(model, **kwargs)

        self.conditional_parameters = []

        if not uniform_parameters or uniform_parameters is None:
            self.uniform_parameters = []
        else:
            self.uniform_parameters = uniform_parameters

        self.conditional_likelihood = conditional_likelihood

        self.conditional = any([self.uniform_parameters,
                                self.conditional_likelihood])

    @property
    def uniform_dims(self):
        """Number of uniform parameters"""
        return len(self.uniform_parameters)

    @property
    def flow_dims(self):
        """Return the number of dimensions for the normalising flow"""
        return self.rescaled_dims - self.uniform_dims

    @property
    def flow_names(self):
        return list(set(self.rescaled_names) - set(self.uniform_parameters))

    @property
    def conditional_dims(self):
        return len(self.conditional_parameters)

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()

        self.set_uniform_parameters()
        self.set_likelihood_parameter()

        self.set_boundary_inversion()

        if self.rescale_parameters:
            # if rescale is a list, there are the parameters to rescale
            # else all parameters are rescale
            if not isinstance(self.rescale_parameters, list):
                self.rescale_parameters = self.rescaled_names.copy()

            self.rescale_parameters = list(set(self.rescale_parameters)
                                           - set(self.uniform_parameters))

            for i, rn in enumerate(self.rescaled_names):
                if rn in self.rescale_parameters:
                    self.rescaled_names[i] += '_prime'

            self._min = {n: self.model.bounds[n][0] for n in self.model.names}
            self._max = {n: self.model.bounds[n][1] for n in self.model.names}
            self._rescale_factor = np.ptp(self.rescale_bounds)
            self._rescale_shift = self.rescale_bounds[0]

            self.rescale = self._rescale_to_bounds
            self.inverse_rescale = self._inverse_rescale_to_bounds
            logger.info(f'Set to rescale inputs to {self.rescale_bounds}')

            if self.update_bounds:
                logger.info(
                    'Rescaling will use min and max of current live points')
            else:
                logger.info('Rescaling will use model bounds')

        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def set_likelihood_parameter(self):
        if self.conditional_likelihood:
            self.likelihood_index = len(self.conditional_parameters)
            self.conditional_parameters += ['logL']
            self.likelihood_distribution = InterpolatedDistribution('logL')

    def set_uniform_parameters(self):
        """
        Set the uniform parameters
        """
        if self.uniform_parameters:
            self.uniform_indices = np.arange(len(self.uniform_parametrs)) \
                                   + len(self.conditional_parameters)
            self.conditional_parameters += self.uniform_parameters
            self.uniform_min = \
                [self.model.bounds[n][0] for n in self.uniform_parameters]
            self.uniform_max = \
                [self.model.bounds[n][1] for n in self.uniform_parameters]

            self._uniform_log_prob = \
                -np.log(np.ptp([self.uniform_min, self.uniform_max]))
        else:
            logger.info('No uniform parameters to set')

    def initialise(self):
        """
        Initialise the proposal class.

        This includes:
            * Setting up the rescaling
            * Verifying the rescaling is invertible
            * Intitialising the FlowModel
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.set_rescaling()
        self.verify_rescaling()
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info('Overwritting fuzz factor with expansion fraction')
            self.fuzz = \
                (1 + self.expansion_fraction) ** (1 / self.flow_dims)
            logger.info(f'New fuzz factor: {self.fuzz}')
        self.flow_config['model_config']['n_inputs'] = self.flow_dims
        if self.conditional:
            self.flow_config['model_config']['kwargs']['context_features'] = \
                    self.conditional_dims
        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def train_on_data(self, x_prime, output):
        """
        Function that takes live points converts to numpy array and calls
        the train function. Live points should be in the X' (x prime) space.
        """
        x_prime_array = live_points_to_array(x_prime, self.flow_names)
        context = self.get_context(x_prime)
        self.train_context(context)
        self.flow.train(x_prime_array, context=context, output=output,
                        plot=self._plot_training)

    def train_context(self, context):
        if self.conditional_likelihood:
            self.likelihood_distribution.update_samples(
                    context[:, self.likelihood_index], reset=True)

    def sample_context_parameters(self, n):
        """
        Draw n samples from the context distributions
        """
        context = np.empty([n, self.conditional_dims])
        log_prob = np.zeros(n,)
        if self.uniform_parameters:
            u, log_prob_u = self.samples_uniform_parameters
            context[:, self.uniform_indices] = u
            log_prob += log_prob_u
        if self.conditional_likelihood:
            context[:, self.likelihood_index] = \
                self.likelihood_distribution.sample(n)

        return context, log_prob

    def sample_uniform_parameters(self, n):
        """
        Draw n parameters from a uniform distribution
        """
        if self.uniform_parameters:
            x = np.random.uniform(self.uniform_min, self.uniform_max,
                                  (n, self.uniform_dims))
            log_prob = self._uniform_log_prob * np.ones(n)
            return x, log_prob
        else:
            return None

    def get_context(self, x):
        """
        Get the context parameters if empty return None
        """
        context = np.empty([x.size, self.conditional_dims])
        if self.uniform_parameters:
            context[:, self.uniform_indices] = \
                live_points_to_array(x, self.uniform_parameters)
        if self.conditional_likelihood:
            context[:, self.likelihood_index] = x['logL'].flatten()

        if context.size:
            return context

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
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            jacobian)
        """
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x, compute_radius=compute_radius)
            log_J += log_J_rescale

        x_flow = live_points_to_array(x, names=self.flow_names)
        context = self.get_context(x)

        if x_flow.ndim == 1:
            x_flow = x_flow[np.newaxis, :]
        if x_flow.shape[0] == 1:
            if self.clip:
                x_flow = np.clip(x_flow, *self.clip)
        z, log_prob = self.flow.forward_and_log_prob(x_flow, context=context)
        return z, log_prob + log_J

    def backward_pass(self, z, context=None, rescale=True, log_prob=0):
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
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            Jacobian)
        """
        if context is None and self.conditional:
            context, log_prob_context = \
                self.sample_context_parameters(z.shape[0])
            log_prob += log_prob_context

        try:
            x_flow, log_prob_flow = self.flow.sample_and_log_prob(
                z=z, context=context, alt_dist=self.alt_dist)
        except AssertionError:
            return np.array([]), np.array([])

        log_prob += log_prob_flow
        if context is not None and self.uniform_parameters:
            x = np.concatenate([x_flow, context[:-1]], axis=1)
        else:
            x = x_flow

        valid = np.isfinite(log_prob)
        x, log_prob = x[valid], log_prob[valid]
        x = numpy_array_to_live_points(
            x.astype(DEFAULT_FLOAT_DTYPE), self.rescaled_names)
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_prob -= log_J
            x, log_prob = self.check_prior_bounds(x, log_prob)
        return x, log_prob


class AugmentedFlowProposal(FlowProposal):

    def __init__(self, model, augment_features=1, **kwargs):
        super().__init__(model, **kwargs)
        self.augment_features = augment_features

    def set_rescaling(self):
        super().set_rescaling()
        self.augment_names = [f'e_{i}' for i in range(self.augment_features)]
        self.names += self.augment_names
        self.rescaled_names += self.augment_names
        logger.info(f'augmented x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(
                f'augmented x prime space parameters: {self.rescaled_names}'
                )

    def initialise(self):
        """
        Initialise the proposal class
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.set_rescaling()

        m = np.ones(self.flow_dims)
        m[-self.augment_features:] = -1
        self.flow_config['model_config']['kwargs']['mask'] = m

        self.flow_config['model_config']['n_inputs'] = self.flow_dims

        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def _rescale_with_bounds(self, x, generate_augment=True):
        """
        Rescale the inputs to [-1, 1] using the bounds as the min and max
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = 0.
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x_prime[rn] = 2 * ((x[n] - self.model.bounds[n][0])
                                   / (self.model.bounds[n][1]
                                      - self.model.bounds[n][0])) - 1

                log_J += np.log(2) - np.log(self.model.bounds[n][1]
                                            - self.model.bounds[n][0])
            elif n not in self.augment_names:
                x_prime[rn] = x[n]
        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']
        if generate_augment:
            if x_prime.size == 1:
                x_prime[self.augment_names] = self.augment_features * (0.,)
            else:
                for an in self.augment_names:
                    x_prime[an] = np.random.randn(x_prime.size)

        return x_prime, log_J

    def _inverse_rescale_with_bounds(self, x_prime):
        """
        Rescale the inputs from the prime space to the phyiscal space
        using the model bounds
        """
        x = np.zeros([x_prime.size], dtype=self.x_dtype)
        log_J = 0.
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x[n] = (self.model.bounds[n][1] - self.model.bounds[n][0]) \
                        * ((x_prime[rn] + 1) / 2) + self.model.bounds[n][0]
                log_J += np.log(self.model.bounds[n][1]
                                - self.model.bounds[n][0]) - np.log(2)
            else:
                x[n] = x_prime[rn]
        x['logP'] = x_prime['logP']
        x['logL'] = x_prime['logL']
        return x, log_J

    def augmented_prior(self, x):
        """
        Log guassian for augmented variables
        """
        logP = 0.
        for n in self.augment_names:
            logP += stats.norm.logpdf(x[n])
        return logP

    def _marginalise_augment(self, x, K=50):
        """
        Marginalise out the augmented feautures
        """
        x_prime, log_J = self.rescale(x, generate_augment=False)
        x_prime = live_points_to_array(x_prime, names=self.rescaled_names)
        x_prime = np.repeat(x_prime, K, axis=0)

        x_prime[:, -self.augment_features:] = np.random.randn(
                x.size * K, self.augment_features)
        z, log_prob = self.flow.forward_and_log_prob(x_prime)
        log_prob = log_prob.reshape(K, x.size)
        log_prob_e = np.sum(stats.norm.logpdf(
            x_prime[:, -self.augment_features:]), axis=-1
            ).reshape(K, x.size)
        assert log_prob.shape == log_prob_e.shape
        log_prob = -np.log(K) + logsumexp(log_prob - log_prob_e, (0))
        return log_prob - log_J

    def log_prior(self, x):
        """
        Compute the prior probability
        """
        physical_prior = self.model.log_prior(x[self.model.names])
        augmented_prior = self.augmented_prior(x)
        # return physical_prior
        return physical_prior + augmented_prior

    def compute_weights(self, x, log_q):
        """
        Compute the weight for a given set of samples
        """
        # log_q = self._marginalise_augment(x)
        log_p = self.log_prior(x)
        x['logP'] = log_p
        x['logL'] = log_q
        log_w = log_p - log_q
        log_w -= np.nanmax(log_w)
        return log_w, log_q

    def populate(self, worst_point, N=10000, plot=True):
        """Populate a pool of latent points"""
        if self.fixed_radius:
            r = self.fixed_radius
        else:
            worst_z, worst_q = self.forward_pass(worst_point, rescale=True)
            # worst_log_q = self._marginalise_augment(worst_point)
            # worst_log_q = worst_q
            r = self.radius(worst_z)
        logger.debug(f'Populating proposal with lantent radius: {r:.5}')
        warn = True
        if not self.keep_samples or not self.indices:
            self.x = np.array([], dtype=self.x_dtype)
            z_samples = np.empty([0, self.dims])
        counter = 0
        zero_counter = 0
        while len(self.x) < N:
            while True:
                z = self.draw_latent_prior(self.dims, r=r, N=self.drawsize,
                                           fuzz=self.fuzz)
                if z.size:
                    break

            x, log_q = self.backward_pass(z, rescale=True)
            # rescale given priors used initially, need for priors
            log_w, log_q = self.compute_weights(x, log_q)
            # x = x[log_q > worst_log_q]
            # log_w = log_w[log_q > worst_log_q]
            log_u = np.log(np.random.rand(x.shape[0]))
            indices = np.where((log_w - log_u) >= 0)[0]

            if not len(indices) or (len(indices) / self.drawsize < 0.001):
                logger.warning(
                    'Rejection sampling produced almost zero samples!')
                zero_counter += 1
                if zero_counter == 10 and self.x.size < (N // 2) and \
                        self.latent_prior != 'uniform':
                    logger.warning(
                        'Proposal is too ineffcient, ''reducing radius')
                    r *= 0.99
                    logger.warning(f'New radius: {r}')
                    self.x = np.array([], dtype=self.x_dtype)
                    z_samples = np.empty([0, self.dims])
                    zero_counter = 0
                    continue
            if len(indices) / self.drawsize < 0.01:
                if warn:
                    logger.warning(
                        'Rejection sampling accepted less than 1 percent of '
                        f'samples! ({len(indices) / self.drawsize})'
                        )
                    warn = False

            # array of indices to take random draws from
            self.x = np.concatenate([self.x, x[indices]], axis=0)
            z_samples = np.concatenate([z_samples, z[indices]], axis=0)
            if counter % 10 == 0:
                logger.debug(f'Accepted {self.x.size} / {N} points')
            counter += 1

        if plot:
            plot_live_points(
                self.x,
                filename=f'{self.output}/pool_{self.populated_count}.png'
                )

        self.samples = self.x[self.model.names + ['logP', 'logL']]

        if self.check_acceptance:
            self.approx_acceptance.append(self.compute_acceptance(worst_q))
            logger.debug('Current approximate acceptance: '
                         f'{self.approx_acceptance[-1]}')
            self.evaluate_likelihoods()
            self.acceptance.append(
                self.compute_acceptance(worst_point['logL']))
            logger.debug(f'Current acceptance {self.acceptance[-1]}')
            if plot:
                plot_acceptance(
                    self.approx_acceptance, self.acceptance,
                    labels=['approx', 'analytic'],
                    filename=f'{self.output}/proposal_acceptance.png'
                    )
        else:
            self.samples['logL'] = np.zeros(self.samples.size,
                                            dtype=self.samples['logL'].dtype)

        self.indices = np.random.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        logger.info(f'Proposal populated with {len(self.indices)} samples')
