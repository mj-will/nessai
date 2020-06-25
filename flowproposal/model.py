
import numpy as np

from .livepoint import parameters_to_live_point, numpy_array_to_live_points

class Model:

    names = [] # Names of parameters, e.g. ['p1','p2']
    dims = 0
    bounds = np.array([])

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
        while (logP == - np.inf):
            p = numpy_array_to_live_points(
                    np.random.uniform(self.bounds.T[0], self.bounds.T[1], [1, self.dims]),
                    self.names)
            logP=self.log_prior(p)
        return p

    def _multiple_new_points(self, N):
        new_points = np.array([], dtype=[(n, 'f') for n in self.names + ['logP', 'logL']])
        while new_points.size < N:
            p = numpy_array_to_live_points(
                    np.random.uniform(self.bounds.T[0], self.bounds.T[1], [N, self.dims]).astype(float),
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

    def header(self):
        """
        Return a string with the output file header
        """
        return '\t'.join(self.names) + '\tlogL'
