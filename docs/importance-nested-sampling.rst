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


Output
======

The importance nested sampler returns:

- a result file, by default :code:`result.h5`,
  - :code:`samples` contains the final samples drawn from meta-proposal,
  - :code:`log_evidence` is the final estimate of the log-evidence,
- a state plot (:code:`state.png`), this is similar to the state plot for the standard sampler,
- a trace plot (:code:`trace.png`), this is similar to the trace plot from the standard sampler but plots the ratio of the prior and the meta-proposal on the x-axis,
- a levels plot (:code:`levels.png`), this shows the log-likelihood distribution for each proposal.
