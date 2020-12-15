======================
Sampler Configuration
======================


The proposal process is managed by a proposal object that inherits from ``Proposal``. By default the sampler starts with *uniformed sampling* were samples are drawn using the ``new_point`` method from the model. Once a specific criteria the sampler switches to using a proposal method which includes a normalising flow (FlowPropolsal). Both stages of the sampling are configure when creating an instance of ``FlowSampler``.

General configuration
=====================

These are general settings which apply to the whole algorithm:

* ``nlive``: number of live points
* ``stopping``: the stopping criteria


For a complete list see :py:class:`nessai.nestedsampler.NestedSampler`


Configuring uninformed proposal
===============================



Configuring FlowProposal
========================
