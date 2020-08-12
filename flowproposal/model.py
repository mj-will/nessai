
import numpy as np

from .livepoint import (
        numpy_array_to_live_points,
        get_dtype,
        DEFAULT_FLOAT_DTYPE
        )


class Model:

    likelihood_evaluations = 0
    names = []    # Names of parameters, e.g. ['p1','p2']
    bounds = {}
    _lower = None
    _upper = None

    @property
    def dims(self):
        return len(self.names)

    @property
    def lower_bounds(self):
        if self._lower is None:
            bounds_array = np.array(list(self.bounds.values()))
            self._lower = bounds_array[:, 0]
            self._upper = bounds_array[:, 1]
        return self._lower

    @property
    def upper_bounds(self):
        if self._upper is None:
            bounds_array = np.array(list(self.bounds.values()))
            self._lower = bounds_array[:, 0]
            self._upper = bounds_array[:, 1]
        return self._upper

    def new_point(self, N=1):
        """
        Create a new LivePoint, drawn from within bounds

        -----------
        Return:
            p: :obj:`cpnest.parameter.LivePoint`
        """
        if N >= 1:
            return self._multiple_new_points(N)
        else:
            return self._single_new_point()

    def _single_new_point(self):
        logP = -np.inf
        while (logP == -np.inf):
            p = numpy_array_to_live_points(
                    np.random.uniform(self.lower_bounds, self.upper_bounds,
                                      [1, self.dims]),
                    self.names)
            logP = self.log_prior(p)
        return p

    def _multiple_new_points(self, N):
        new_points = np.array([], dtype=get_dtype(self.names,
                                                  DEFAULT_FLOAT_DTYPE))
        while new_points.size < N:
            p = numpy_array_to_live_points(
                    np.random.uniform(self.lower_bounds, self.upper_bounds,
                                      [N, self.dims]),
                    self.names)
            logP = self.log_prior(p)
            new_points = np.concatenate([new_points, p[np.isfinite(logP)]])
        return new_points

    def log_likelihood(self, x):
        """
        returns log likelihood of given parameter

        ------------
        Parameter:
            param: :obj:`cpnest.parameter.LivePoint`
        """
        pass

    def log_prior(self, x):
        """
        Returns log of prior.
        Default is flat prior within bounds

        ----------
        Parameter:
            param: :obj:`cpnest.parameter.LivePoint`

        ----------
        Return:
            0 if param is in bounds
            -np.inf otherwise
        """
        pass

    def evaluate_log_likelihood(self, x):
        self.likelihood_evaluations += 1
        return self.log_likelihood(x)

    def header(self):
        """
        Return a string with the output file header
        """
        return '\t'.join(self.names) + '\tlogL'
