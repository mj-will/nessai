# -*- coding: utf-8 -*-
"""
Distributions for use with nessai
"""
import logging

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


class InterpolatedDistribution:
    """
    Object the approximates the CDF and inverse CDF
    of a distribution given samples.

    Parameters
    ----------
    names : str
        Name for the parmeter
    samples : array_like, optional
        Initial array of samples to use for interpolation
    """
    def __init__(self, name, samples=None, rescale=False):
        logger.debug(f'Initialising interpolated dist for: {name}')
        self.name = name
        self._cdf_interp = None
        self._inv_cdf_interp = None
        self.samples = None
        self.min = None
        self.max = None
        self.rescale = rescale
        if samples is not None:
            self.update_dist(samples, reset=True)

    def update_samples(self, samples, reset=False, **kwargs):
        """
        Update the samples used for the interpolation

        Parameters
        ----------
        samples : array_like
            Samples used for the update
        reset : bool, optional
            If True new samples are used to replace previous samples.
            If False samples are added to existing samples
        **kwargs
            Arbitrary keyword arguments parsed to scipy.interpolate.splrep
        """
        if samples.ndim > 1:
            raise RuntimeError('Samples must be a 1-dimensional array')
        if reset or self.samples is None:
            self.samples = np.unique(samples)
            self.min = self.samples[0]
            self.max = self.samples[-1]
        else:
            self.samples = \
                np.unique(np.concatenate([self.samples, samples], axis=-1))
        cdf = np.arange(self.samples.size) / (self.samples.size - 1)
        assert self.samples.size == cdf.size
        self._cdf_interp = interpolate.splrep(self.samples, cdf, **kwargs)
        self._inv_cdf_interp = interpolate.splrep(cdf, self.samples, **kwargs)

    def cdf(self, x, **kwargs):
        """
        Compute the interpolated CDF

        Parameters
        ----------
        x : array_like
            Samples to compute CDF for
        **kwargs
            Arbitrary keyword arguments parsed to scipy.interpolate.splev

        Returns
        -------
        array_like
            Values of the CDF for each sample in x
        """
        return interpolate.splev(x, self._cdf_interp, **kwargs)

    def inverse_cdf(self, u, **kwargs):
        """
        Compute the interpolated inverse CDF

        Parameters
        ----------
        x : array_like
            Samples to compute the inverse CDF for
        **kwargs
            Arbitrary keyword arguments parsed to scipy.interpolate.splev

        Returns
        -------
        array_like
            Values of the inverse CDF for each sample in x
        """
        return interpolate.splev(u, self._inv_cdf_interp, **kwargs)

    def sample(self, n=1, min_logL=None, **kwargs):
        """
        Draw a sample from the approximated distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to draw
        **kwargs
           Arbitrary keyword arguments parsed to `inverse_cdf`

        Returns
        -------
        array_like
            Array of n samples drawn from the interpolate distribution
        """
        if min_logL is not None and min_logL > self.min:
            u = np.random.uniform(max(0, self.cdf(min_logL)), 1, n)
        else:
            u = np.random.rand(n)
        if not self.rescale:
            return self.inverse_cdf(u, **kwargs)
        else:
            return ((self.inverse_cdf(u, **kwargs) - self.min) /
                    (self.max - self.min))


class CategoricalDistribution:
    """Distribution for handling discrete conditional parameters.


    Parameters
    ----------
    n : int
        Number of discrete parameters
    """
    def __init__(self, n=None, classes=None, samples=None):
        self.n = n
        self.samples = None
        self.p = None
        self.classes = sorted(classes) if classes is not None else None

        if samples is not None:
            self.update_samples(samples)

    def update_samples(self, samples, reset=False):
        """Update the samples used to determine the distribution
        Parameters
        ----------
        samples : array_like
            Samples used for the update
        reset : bool, optional
            If True new samples are used to replace previous samples.
            If False samples are added to existing samples
        """
        samples = np.squeeze(np.array(samples))
        if samples.ndim > 1:
            raise RuntimeError('Samples must be a 1-dimensional array')
        classes = np.unique(samples).tolist()

        if self.classes is None:
            self.classes = classes
            logger.info(f'Found classes: {classes}')
        elif not np.isin(classes, self.classes).all():
            raise RuntimeError(
                f'New samples contain different classes: {classes}. '
                f'Expected {self.classes}'
            )

        if self.n is None:
            self.n = len(classes)
        elif len(classes) > self.n:
            raise RuntimeError(
                f'Categorical distribution has {self.n} classes, '
                f'{len(classes)} given.'
            )

        if reset or self.samples is None:
            self.samples = samples
            logger.debug('Replacing existing samples')
        else:
            logger.debug('Adding to existing samples')
            self.samples = np.concatenate([self.samples, samples], axis=-1)

        unique, counts = np.unique(self.samples, return_counts=True)
        self.p = self.n * [0]
        for u, c in zip(unique, counts):
            self.p[self.classes.index(u)] = c / self.samples.size

        logger.info(f'New probabilities ({self.classes}): {self.p}')

    def log_prob(self, samples):
        """Compute the log-probablity of the samples.

        Parameters
        ----------
        samples : :obj:`numpy.ndarray`
            Array of samples.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of probabilties.
        """
        log_prob = np.zeros(samples.size)
        for c, p in zip(self.classes, self.p):
            log_prob[(samples == c).flatten()] = np.log(p)
        return log_prob

    def sample(self, n=1):
        """Draw a new sample(s) from the categorical distribution.

        Parameters
        ----------
        n :  int, optional
            Number of samples to draw.
        """
        samples = np.random.choice(self.classes, size=(n, 1), p=self.p)
        log_prob = self.log_prob(samples)
        return samples, log_prob
