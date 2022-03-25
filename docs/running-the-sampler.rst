===================
Running the sampler
===================


Defining the model
==================

The user must define a model that inherits from :py:class:`nessai.model.Model` that defines two parameters and two methods. This object contains the Bayesian prior and likelihood which will be used for sampling.

**Parameters:**

- ``names``: a ``list`` of ``str`` with names for the parameters to be sampled
- ``bounds``: a ``dict`` with a tuple for each parameter in ``names`` with defines the lower and upper bounds of the priors.

**Methods:**

The user MUST define these two methods, the input to both is a structured numpy array with fields defined by ``names``.

- ``log_prior``: return the log-prior probability of a live point (and enforce the bounds)
- ``log_likelihood``: return the log-likelihood probability of a live point (must be finite)

The input to both methods is a live point ``x`` which is an instance of a structured numpy array with one field for each parameters in ``names`` and two additional fields ``logP`` and ``logL``. Each parameter can be accessed using the name of each field like you would a dictionary.

For examples of using live points see: :ref:`using live points<Using live points>`

Example model
-------------

Here's an example of a simple model taken from one of the examples:

.. literalinclude:: ../examples/2d_gaussian.py
    :language: python
    :pyobject: GaussianModel


Initialising and running the sampler
====================================

Once a modelled is defined, create an instance of :py:class:`nessai.flowsampler.FlowSampler`. This is when the sampler and the proposal methods are configured, for example setting the number of live points (``nlive``) or setting the class of normalising flow to use. ``nessai`` includes a large variety of settings that control different aspects of the sampler, these can be essential to efficient sampling. See :doc:`sampler configuration<sampler-configuration>` for an in-depth explanation of all the settings.

.. code-block:: python

    from nessai.flowsampler import FlowSampler

    # Initialise sampler with the model
    sampler = FlowSampler(GaussianModel(), output='./', nlive=1000)
    # Run the sampler
    sampler.run()


Sampler output
==============

Once the sampler has converged the results and other automatically generated plots with be saved in the directory specified as ``output``. By default this will include:

* ``result.json``: a json file which contains various fields, the most relevant of which are ``posterior_samples``, ``log_evidence`` and ``information``.
* ``posterior_distribution.png``: a corner-style plot of the posterior distribution.
* ``trace.png``: a trace plot which shows the nested samples for each sampled parameter for the entire sampling process.
* ``state.png``: the *state* plot which shows various statistics tracked by the sampler.
* ``insertion_indices.png``: the distribution of insertion indices for all of the nested samples.
* ``logXlogL.png``: the evolution of the maximum log-likelihood versus the log prior volume.
* two resume files (``.pkl``) used for resuming the sampler.
* ``config.json``: the exact configuration used for the sampler.

For a more detail explanation of outputs and examples, see :ref:`here<Detailed explanation of outputs>`


Complete examples
=================

For complete examples see :doc:`gaussian-example` and the `example directory <https://github.com/mj-will/nessai/tree/master/examples>`_.
