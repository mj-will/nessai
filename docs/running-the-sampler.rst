===================
Running the sampler
===================


Defining the model
==================

The user must define a model that inherits from ``nessai.model.Model`` that defines two parameters and two methods.

**Parameters:**

- ``names``: a ``list`` of ``str`` with names for the parameters to be sampled
- ``bounds``: a ``dict`` with a tuple for each parameter in ``names`` with defines the lower and upper bounds

**Methods:**

The user MUST define these two methods, the inputs to both is a structured numpy array with fields defined by ``names``.

- ``log_prior``: return the log-prior probability of a live point (and enforce the bounds)
- ``log_likelihood``: return the log-likelihood probability of a live point (must be finite)

The input to both methods are a live point `x` which is an instance of a structured numpy array with one field for each parameters in ``names`` and two additional fields ``logP`` and ``logL``. Each parameter can be accessed using the name of each field like you would a dictionary.

**Using live points:**

Here's an example of using the live points:

.. code-block:: python

    >>> from nessai.livepoint import parameters_to_live_point
    >>> x = parameters_to_live_point([1, 2], ['a', 'b'])
    >>> print(x)    # the live point
    (1., 2., 0., 0.)
    >>> print(x.dtype.names)
    ('a', 'b', 'logP', 'logL')
    >>> print(x['a'])    # the value of parameter 'a'
    1.0


The structured arrays used for live points can also contain multiple live points:

.. code-block:: python

    >>> from nessai.livepoint import numpy_array_to_live_points
    >>> x = np.random.rand(10, 2)    # 10 live points with 2 parameters
    >>> print(x)
    [[0.72451217 0.1788154 ]
     [0.31549832 0.55898106]
     [0.74000576 0.73103116]
     [0.37362176 0.25791271]
     [0.61056168 0.05940721]
     [0.33988486 0.54106604]
     [0.82653691 0.14523437]
     [0.62390321 0.32606928]
     [0.21743918 0.23915047]
     [0.45478996 0.09699358]]
    >>> y = numpy_array_to_live_points(x, ['a', 'b'])    # call the paramters a and b
    >>> print(y)
    array([(0.72451217, 0.1788154 , 0., 0.), (0.31549832, 0.55898106, 0., 0.),
           (0.74000576, 0.73103116, 0., 0.), (0.37362176, 0.25791271, 0., 0.),
           (0.61056168, 0.05940721, 0., 0.), (0.33988486, 0.54106604, 0., 0.),
           (0.82653691, 0.14523437, 0., 0.), (0.62390321, 0.32606928, 0., 0.),
           (0.21743918, 0.23915047, 0., 0.), (0.45478996, 0.09699358, 0., 0.)],
          dtype=[('a', '<f8'), ('b', '<f8'), ('logP', '<f8'), ('logL', '<f8')])
    >>> y['a']    # all the values of a
    array([0.72451217, 0.31549832, 0.74000576, 0.37362176, 0.61056168,
           0.33988486, 0.82653691, 0.62390321, 0.21743918, 0.45478996])
    >>> y[0]      # the first live point
    (0.72451217, 0.1788154, 0., 0.)



Example model
-------------

Here's an example of what a model could look like:

.. code-block:: python

    class GaussianModel(Model):

        def __init__(self):
            self.names = ['x', 'y']
            self.bounds = {'x': [-10, 10], 'y': [-10, 10]}

        def log_prior(self, x):
            """
            Returns log of prior.
            """
            log_p = 0.
            # Iterate through each parameter (x and y)
            # since the live points are a structured array we can
            # get each value using just the name
            for i, n in enumerate(self.names):
                log_p += xlogy(1, (x[n] >= self.bounds[n][0])
                               & (x[n] <= self.bounds[n][1]))
                - xlogy(1, self.bounds[n][1] - self.bounds[n][0])
            return log_p

        def log_likelihood(self, x):
            """
            Returns log likelihood of given parameter
            """
            log_l = 0
            # Use a Guassian pdf and iterate through the parameters
            for pn in self.names:
                log_l += norm.logpdf(x[pn])
            return log_l


Optional methods
----------------

There are further methods that the user can re-define:

**Drawing new points**

The ``Model`` includes a method for drawing new samples ``new_point`` and computing the corresponding log-probability of drawing the sample ``new_point_log_prob``. This is used when populating the initial pool of points and during the inital *uninformed sampling* when the normalising flow is not used. By default these points are drawn uniformally within user-defined ``bounds`` but the user can choose to re-define these methods.

If the samples produced by ``new_point`` are drawn directly from the prior the sampler can be ran with the flag ``analytic_priors=True`` and this will improve the efficiency of the initial *uniformed sampling*.

NOTE: the the new samples returned by ``new_point`` must be a structured array with the correct fields, the functions in ``nessia.livepoint`` are useful here.


Initialising and running the sampler
====================================

sampler = FlowSampler(Gaussian(), output=output, resume=False, nlive=1000,
                      plot=True, flow_config=flow_config, training_frequency=None,
                      maximum_uninformed=1000, rescale_parameters=True, seed=1234)
