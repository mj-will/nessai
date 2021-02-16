#####################
Sampler Configuration
#####################

There are various settings in :code:`nessai` which can be configured. These can be grouped in to general settings and proposal settings. The former controls general aspects of the sampler such as the model being sampler or how many live points are used. The latter affect the proposal process and how new points are drawn.

All of the settings are controled when creating an instance of :class:`~nessai.flowsampler.FlowSampler`.

*********************
General configuration
*********************

These are general settings which apply to the whole algorithm and are parsed to :mod:`~nessai.nestedsampler.NestedSampler`. However some of these settings, such as :code:`training_frequency` which defines how often the proposal method is retrained.

.. autoclass:: nessai.nestedsampler.NestedSampler
    :members: None


**********************
Proposal configuration
**********************

The proposal configuration includes a variety

.. autoclass:: nessai.proposal.FlowProposal
    :members: None
