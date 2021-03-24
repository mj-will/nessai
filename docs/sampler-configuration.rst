=====================
Sampler Configuration
=====================

There are various settings in ``nessai`` which can be configured. These can be grouped in to general settings and proposal settings. The former controls general aspects of the sampler such as the model being sampler or how many live points are used. The latter affect the proposal process and how new points are drawn.

All of the settings are controled when creating an instance of :class:`~nessai.flowsampler.FlowSampler`. The most important settings to consider when using ``nessai`` are the :doc:`reparameterisations<reparameterisations>` used for the proposals.

General configuration
=====================

These are general settings which apply to the whole algorithm and are parsed to :mod:`~nessai.nestedsampler.NestedSampler`. However some of these settings, such as :code:`training_frequency` which defines how often the proposal method is retrained.

.. autoapiclass:: nessai.nestedsampler.NestedSampler
    :members: None


Proposal configuration
======================

The proposal configuration includes a variety of settings ranging from the hyper-parameters for the normalising flow to the size of pool used to store new samples. This also includes the reparameterisations which are essential to efficient sampling. All the available settings are listed below and there are dedicated pages that explain how to configure the :doc:`reparmeterisations<reparameterisations>` and :doc:`normalising
flows<normalising-flows-configuration>`.

.. autoapiclass:: nessai.proposal.flowproposal.FlowProposal
    :members: None
