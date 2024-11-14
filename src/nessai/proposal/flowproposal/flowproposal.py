"""Proposal class that uses normalising flows and rejection sampling.

This is the default proposal used by nessai.
"""

import datetime
import logging
from functools import partial

import numpy as np
from scipy.special import logsumexp

from ... import config
from ...livepoint import (
    empty_structured_array,
    numpy_array_to_live_points,
)
from ...utils import (
    compute_radius,
    get_uniform_distribution,
)
from ...utils.sampling import NDimensionalTruncatedGaussian
from ...utils.structures import get_subset_arrays
from .base import BaseFlowProposal

logger = logging.getLogger(__name__)


class FlowProposal(BaseFlowProposal):
    """
    Object that handles training and proposal points

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User defined model.
    latent_prior : {'truncated_gaussian', 'gaussian', 'uniform_nsphere', \
            'gaussian'}, optional
        Prior distribution in the latent space. Defaults to
        'truncated_gaussian'.
    poolsize : int, optional
        Size of the proposal pool. Defaults to 10000.
    drawsize : int, optional
        Number of points to simultaneously draw when populating the proposal
        Defaults to 10000
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
    """

    def __init__(
        self,
        model,
        poolsize=None,
        latent_prior="truncated_gaussian",
        constant_volume_mode=True,
        volume_fraction=0.95,
        fuzz=1.0,
        fixed_radius=False,
        drawsize=None,
        truncate_log_q=False,
        expansion_fraction=4.0,
        min_radius=False,
        max_radius=50.0,
        compute_radius_with_all=False,
        **kwargs,
    ):
        super().__init__(
            model,
            poolsize=poolsize,
            **kwargs,
        )
        logger.debug("Initialising FlowProposal")

        self._draw_func = None
        self._populate_dist = None

        self.configure_population(
            drawsize,
            fuzz,
            expansion_fraction,
            latent_prior,
        )

        self.truncate_log_q = truncate_log_q
        self.constant_volume_mode = constant_volume_mode
        self.volume_fraction = volume_fraction

        self.compute_radius_with_all = compute_radius_with_all
        self.configure_fixed_radius(fixed_radius)
        self.configure_min_max_radius(min_radius, max_radius)

        self.configure_latent_prior()
        self.alt_dist = None

    def configure_population(
        self,
        drawsize,
        fuzz,
        expansion_fraction,
        latent_prior,
    ):
        """
        Configure settings related to population
        """
        if drawsize is None:
            drawsize = self.poolsize

        self.drawsize = drawsize
        self.fuzz = fuzz
        self.expansion_fraction = expansion_fraction
        self.latent_prior = latent_prior

    def configure_latent_prior(self):
        """Configure the latent prior"""
        if self.latent_prior == "truncated_gaussian":
            from ...utils import draw_truncated_gaussian

            self._draw_latent_prior = draw_truncated_gaussian

        elif self.latent_prior == "gaussian":
            logger.warning("Using a gaussian latent prior WITHOUT truncation")
            from ...utils import draw_gaussian

            self._draw_latent_prior = draw_gaussian
        elif self.latent_prior == "uniform":
            from ...utils import draw_uniform

            self._draw_latent_prior = draw_uniform
        elif self.latent_prior in ["uniform_nsphere", "uniform_nball"]:
            from ...utils import draw_nsphere

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

    def set_rescaling(self):
        """Set the rescaling functions.

        Also configures constant volume mode since this must be done AFTER
        the reparameterisations are configured.
        """
        super().set_rescaling()
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info("Overwriting fuzz factor with expansion fraction")
            self.fuzz = (1 + self.expansion_fraction) ** (
                1 / self.rescaled_dims
            )
            logger.info(f"New fuzz factor: {self.fuzz}")
        self.configure_constant_volume()

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

    def prep_latent_prior(self):
        """Prepare the latent prior."""
        if self.latent_prior == "truncated_gaussian":
            self._populate_dist = NDimensionalTruncatedGaussian(
                self.dims,
                self.r,
                fuzz=self.fuzz,
                rng=self.rng,
            )
            self._draw_func = self._populate_dist.sample
        elif self.latent_prior == "flow":
            self._draw_func = lambda N: self.flow.sample_latent_distribution(N)
        else:
            assert self.rng is not None
            self._draw_func = partial(
                self._draw_latent_prior,
                dims=self.dims,
                r=self.r,
                fuzz=self.fuzz,
                rng=self.rng,
            )

    def draw_latent_prior(self, n):
        """Draw n samples from the latent prior."""
        return self._draw_func(N=n)

    def backward_pass(
        self,
        z,
        rescale=True,
        discard_nans=True,
        return_z=False,
        return_unit_hypercube=False,
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
        except AssertionError as e:
            logger.warning(
                "Assertion error raised when sampling from the flow."
                f"Error: {e}"
            )
            if return_z:
                return np.array([]), np.array([]), np.array([])
            else:
                return np.array([]), np.array([])

        if discard_nans:
            valid = np.isfinite(log_prob)
            x, log_prob, z = get_subset_arrays(valid, x, log_prob, z)
        x = numpy_array_to_live_points(
            x.astype(config.livepoints.default_float_dtype),
            self.prime_parameters,
        )
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(
                x, return_unit_hypercube=return_unit_hypercube
            )
            # Include Jacobian for the rescaling
            log_prob -= log_J
            if not return_unit_hypercube:
                x, z, log_prob = self.check_prior_bounds(x, z, log_prob)
        if return_z:
            return x, log_prob, z
        else:
            return x, log_prob

    def populate(
        self,
        worst_point,
        n_samples=10000,
        plot=True,
        r=None,
        max_samples=1_000_000,
    ):
        """
        Populate a pool of latent points given the current worst point.

        Parameters
        ----------
        worst_point : structured_array
            The current worst point used to compute the radius of the contour
            in the latent space.
        n_samples : int, optional (10000)
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
            samples = empty_structured_array(
                n_samples, dtype=self.population_dtype
            )

        self.prep_latent_prior()

        log_n = np.log(n_samples)
        log_n_expected = -np.inf
        n_proposed = 0
        log_weights = np.empty(0)
        log_constant = -np.inf
        n_accepted = 0
        accept = None

        while n_accepted < n_samples:
            z = self.draw_latent_prior(self.drawsize)
            n_proposed += z.shape[0]

            x, log_q = self.backward_pass(
                z,
                rescale=not self.use_x_prime_prior,
                return_unit_hypercube=self.map_to_unit_hypercube,
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
                    n_samples,
                )

                # Only try rejection sampling if we expected to accept enough
                # points. In the case where we don't, we continue drawing
                # samples
                if log_n_expected >= log_n:
                    log_u = np.log(self.rng.random(len(log_weights)))
                    accept = (log_weights - log_constant) > log_u
                    n_accepted = np.sum(accept)
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break

            else:
                log_w -= log_w.max()
                log_u = np.log(self.rng.random(len(log_w)))
                accept = log_w > log_u
                n_accept_batch = accept.sum()
                m = min(n_samples - n_accepted, n_accept_batch)
                samples[n_accepted : n_accepted + m] = x[accept][:m]
                n_accepted += n_accept_batch
                logger.debug("n accepted: %s / %s", n_accepted, n_samples)

        if self.accumulate_weights:
            if accept is None or len(accept) != len(samples):
                log_u = np.log(self.rng.random(len(log_weights)))
                accept = (log_weights - log_constant) > log_u
            logger.debug("Total number of samples: %s", samples.size)
            n_accepted = np.sum(accept)
            self.x = samples[accept][:n_samples]
        else:
            self.x = samples[:n_samples]

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

        self.indices = self.rng.permutation(self.samples.size).tolist()
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

    def reset(self):
        """Reset the proposal"""
        super().reset()
        self.r = np.nan
        self.alt_dist = None
        self._populate_dist = None
        self._draw_func = None
        self.alt_dist = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_draw_func"] = None
        state["_populate_dist"] = None
        state["alt_dist"] = None
        return state
