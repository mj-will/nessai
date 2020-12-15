===================
Running the sampler
===================


Defining the model
==================

The user must define a model that inherits from :py:class:`nessai.model.Model` that defines two parameters and two methods.

**Parameters:**

- ``names``: a ``list`` of ``str`` with names for the parameters to be sampled
- ``bounds``: a ``dict`` with a tuple for each parameter in ``names`` with defines the lower and upper bounds

**Methods:**

The user MUST define these two methods, the inputs to both is a structured numpy array with fields defined by ``names``.

- ``log_prior``: return the log-prior probability of a live point (and enforce the bounds)
- ``log_likelihood``: return the log-likelihood probability of a live point (must be finite)

The input to both methods are a live point `x` which is an instance of a structured numpy array with one field for each parameters in ``names`` and two additional fields ``logP`` and ``logL``. Each parameter can be accessed using the name of each field like you would a dictionary.

For examples of using live points see: :doc:`using-livepoints`

Example model
-------------

Here's an example of what a model could look like:

.. code-block:: python

    class GaussianModel(Model):
        """
        A simple two-dimensional Guassian likelihood
        """
        def __init__(self):
            self.names = ['x', 'y']
            self.bounds = {'x': [-10, 10], 'y': [-10, 10]}

        def log_prior(self, x):
            """
            Returns log of prior given a live point.
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
            Returns log likelihood of given live point
            """
            log_l = 0
            # Use a Guassian pdf and iterate through the parameters
            for pn in self.names:
                log_l += norm.logpdf(x[pn])
            return log_l




Initialising and running the sampler
====================================

Once a modelled is defined, create an instance of :py:class:`nessai.flowsampler.FlowSampler`. This when te sampler and the proposal methods are configured, for example setting the number of live points or setting the class of normalising flow to use. See :doc:`sampler-configuration` for an in-depth explanation of all the settings.

.. code-block:: python

    sampler = FlowSampler(Gaussian(), output=output)

    sampler.run()
