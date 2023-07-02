=========================
Importance nested sampler
=========================

In `Williams et al. 2023 <https://arxiv.org/abs/2302.08526>`_, we proposed importance nested sampling with normalising flows and called the algorithm i-nessai.
The algorithm described there is available in :code:`nessai` alongside the standard algorithm.

.. important::
    The API for the importance nested sampler is still under-development and may change between minor versions.

Basic usage
===========

The importance nested sampler is implemented in :py:class:`~nessai.samplers.importancesampler.ImportanceNestedSampler`
and, just like standard nessai, it can be run through the :py:class:`~nessai.flowsampler.FlowSampler` class.
To do so, simply set :code:`importance_nested_sampler=True` when creating the instance of :py:class:`~nessai.flowsampler.FlowSampler`.
The sampler is then run using the :py:meth:`nessai.flowsampler.FlowSampler.run`, e.g.

.. code-block:: python

    from nessai.flowsampler import FlowSampler

    sampler = FlowSampler(
        GaussianModel(),
        output="output/",
        nlive=5000,
        importance_nested_sampler=True,
    )
    sampler.run()


.. important::
    The importance nested sampler requires the user to define the methods :code:`to_unit_hypercube` and :code:`from_unit_hypercube` in the model class,
    see the examples linked below.


Configuration
-------------

All keyword arguments passed to  :py:class:`~nessai.flowsampler.FlowSampler` will in turn be passed to either
:py:class:`~nessai.samplers.importancesampler.ImportanceNestedSampler` or :py:class:`~nessai.proposal.importance.ImportanceFlowProposal`.
We now highlight some key settings, for a complete list see the documentation for the two classes,

* :code:`nlive`: more complex problems will often need larger :code:`nlive`,
* :code:`threshold_kwargs`: dictionary of keyword arguments for determining the likelihood threshold, the most important is :code:`q` (:math:`\rho` in the paper),
* :code:`reparameterisation`: either :code:`'logit'` or :code:`None`, the former should be used in most cases, unless the flow has the domain on :math:`[0, 1]^n`,
* :code:`flow_config`: identical to the standard sampler, see :ref:`normalising flows configuration` for details,
* :code:`reset_flow`: reset the flow before training every ith iteration,


Stopping criterion
------------------

By default, the stopping criterion is the log-ratio of the evidence above the likelihood threshold and the current evidence, called :code:`ratio` in the code.
The default tolerance is :code:`tolerance=0.0` and should be suitable for most applications.

Logging
=======

By default the sampler will log every iteration and the log will look something like this:

.. code-block:: console

    07-01 08:46 nessai.samplers.importancesampler INFO    : Removing 1830/3000 samples to train next proposal
    07-01 08:46 nessai.samplers.importancesampler INFO    : Log-likelihood threshold: -487.90392013926345
    07-01 08:46 nessai.samplers.importancesampler INFO    : Training next proposal with 1170 samples
    07-01 08:46 nessai.samplers.importancesampler INFO    : Drawing 2000 samples from the new proposal
    07-01 08:46 nessai.samplers.importancesampler INFO    : Stopping criteria (['ratio']): [0.638027881338866] - Tolerance: [0.0]
    07-01 08:46 nessai.samplers.importancesampler INFO    : Update 1 - log Z: -5.804 +/- 0.089 ESS: 123.7 logL min: -3874.530 logL median: -96.682 logL max: -0.015

From top to bottom this shows:

* the number of samples discarded to train the next proposal (flow),
* the corresponding log-likelihood threshold,
* the start of the training stage with n samples,
* the number of samples being drawn from the new proposal,
* the current values of all stopping criteria and the corresponding tolerance,
* a summary of statistics at the end of the current iteration, this shows:

    * the log-evidence,
    * the effective sample size of the posterior,
    * the minimum, median and maximum log-likelihood of the current set of *live* samples.


Output
======

The importance nested sampler returns:

* a result file, by default :code:`result.h5`,

    * :code:`samples` contains the final samples drawn from meta-proposal,
    * :code:`log_evidence` is the final estimate of the log-evidence,

* a state plot (:code:`state.png`), this is similar to the state plot for the standard sampler,
* a trace plot (:code:`trace.png`), this is similar to the trace plot from the standard sampler but plots the ratio of the prior and the meta-proposal on the x-axis,
* a levels plot (:code:`levels.png`), this shows the log-likelihood distribution for each proposal.


Examples
========

For basic examples, see the `examples directory <https://github.com/mj-will/nessai/tree/inessai-docs/examples/importance_nested_sampler>`_.


Gravitational-wave inference
=============================

The importance nested sampler is not currently supported in :code:`bilby` but will be in a future release.
