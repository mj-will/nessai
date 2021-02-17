import datetime
import logging

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
    def __init__(self, model, n_pool=None):
        self.model = model
        self.populated = True
        self._initialised = False
        self.training_count = 0
        self.population_acceptance = None
        self.population_time = datetime.timedelta()
        self.logl_eval_time = datetime.timedelta()
        self.r = None
        self.n_pool = n_pool
        self.samples = []
        self.indices = []
        self.pool = None

    @property
    def initialised(self):
        """Boolean that indicates if the proposal is initialised or not."""
        return self._initialised

    @initialised.setter
    def initialised(self, boolean):
        """Setter for initialised

        If value is set to true, the proposal method is tested with `test_draw`
        """
        if boolean:
            self._initialised = boolean
            # TODO: make this useable
            # self.test_draw()
        else:
            self._initialised = boolean

    def initialise(self):
        """
        Initialise the proposal
        """
        self.initialised = True

    def configure_pool(self):
        """
        Configure the multiprocessing pool
        """
        if self.pool is None and self.n_pool is not None:
            if hasattr(self, 'check_acceptance') and not self.check_acceptance:
                self.check_acceptance = True
            logger.info(
                f'Starting multiprocessing pool with {self.n_pool} processes')
            import multiprocessing
            self.pool = multiprocessing.Pool(
                processes=self.n_pool,
                initializer=_initialize_global_variables,
                initargs=(self.model,)
            )
        else:
            logger.info('n_pool is none, no multiprocessing pool')

    def close_pool(self, code=None):
        """
        Close the the multiprocessing pool
        """
        if getattr(self, "pool", None) is not None:
            logger.info("Starting to close worker pool.")
            if code == 2:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Finished closing worker pool.")

    def evaluate_likelihoods(self):
        """
        Evaluate the likelihoods for the pool of live points.

        If the multiprocessing pool has been started, the samples will be map
        using `pool.map`.
        """
        st = datetime.datetime.now()
        if self.pool is None:
            for s in self.samples:
                s['logL'] = self.model.evaluate_log_likelihood(s)
        else:
            self.samples['logL'] = self.pool.map(_log_likelihood_wrapper,
                                                 self.samples)
            self.model.likelihood_evaluations += self.samples.size

        self.logl_eval_time += (datetime.datetime.now() - st)

    def draw(self, old_param):
        """
        New a new point given the old point
        """
        raise NotImplementedError

    def test_draw(self):
        """
        Test the draw method to ensure it returns a sample in the correct
        format and the the log prior is computed.
        """
        logger.debug(f'Testing {self.__class__.__name__} draw method')

        test_point = self.model.new_point()
        new_point = self.draw(test_point)

        if new_point['logP'] != self.model.log_prior(new_point):
            raise RuntimeError('Log prior of new point is incorrect!')

        logger.debug(f'{self.__class__.__name__} passed draw test')

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
        state['pool'] = None
        del state['model']
        return state
