==============================
Standard sampler configuration
==============================

.. important::
    Some of settings discussed here only apply to standard ``nessai`` not ``i-nessai``. For ``i-nessai`` see :ref:`Importance Nested Sampler`

There are various settings in ``nessai`` which can be configured. These can be grouped in to general settings and proposal settings. The former controls general aspects of the sampler such as the model being sampler or how many live points are used. The latter affect the proposal process and how new points are drawn.

All of the settings are controlled when creating an instance of :py:class:`~nessai.flowsampler.FlowSampler`. The most important settings to consider when using ``nessai`` are the :doc:`reparameterisations<reparameterisations>` used for the proposals.

Key settings
============

The most important settings to consider when using ``nessai`` with the default :code:`FlowProposal` are:

- :code:`reset_flow` (default :code:`False`): Whether to reset the normalising flow after each time it is trained. If an integer is specified, the flow is reset after every nth time it is trained. This becomes increasingly important for high dimensional problems or problems where the shape of the likelihood changes significantly as the sampling progresses. We recommend trying values between 1 and 16.
- :code:`volume_fraction` (default :code:`0.95`): Fractional value used to truncated the normalising flow latent space when drawing new samples. Lower values are more prone to over-constraining contours whilst higher values can lead to inefficient sampling. If the diagnostics indicate the results are over-constrained, increasing this value may help. We recommend trying values between 0.95 and 0.99.
- :code:`nlive` (default :code:`2000`): Number of live points to use. Increasing this value will lead to more accurate evidence estimates and better exploration of complex, high-dimensional posteriors, but will also increase the runtime. It can also reduce over-constraining since the flow has more sample to learn from.

General configuration
=====================

These are general settings which apply to the whole algorithm and are parsed to :py:class:`~nessai.nestedsampler.NestedSampler`. However some of these settings, such as :code:`training_frequency` which defines how often the proposal method is retrained, will affect the normalising flow used in the proposal class.

.. autoapiclass:: nessai.nestedsampler.NestedSampler
    :members: None


Proposal configuration
======================

The proposal configuration includes a variety of settings ranging from the hyper-parameters for the normalising flow to the size of pool used to store new samples. This also includes the reparameterisations which are essential to efficient sampling. All the available settings are listed below and there are dedicated pages that explain how to configure the :doc:`reparmeterisations<reparameterisations>` and :doc:`normalising
flows<normalising-flows-configuration>`.

.. autoapiclass:: nessai.proposal.flowproposal.FlowProposal
    :members: None

Other proposals
---------------

``nessai`` also includes variations on the main :code:`FlowProposal` class:

- :py:class:`nessai.gw.proposal.GWFlowProposal` as version of :code:`FlowProposal` that includes specific reparameterisations for gravitational-wave inference.
- :py:class:`nessai.proposal.augmented.AugmentedFlowProposal` this proposal is designed for highly multimodal likelihoods where the standard proposal can break down. It is based around using *Augmented Normalising Flows* which introduce extra *augment* dimensions. See the documentation for further details.
