=======
Plugins
=======

Plugins (or entry points) enable the user to define custom classes and
functions that can be used in ``nessai`` without having to edit the source
code. For example, they can be used to support new proposal classes.

``nessai`` current supports the following entry points:

* ``nessai.proposals``: this provides a means for adding additional proposal classes that can be used with the standard nested sampler instead of the default :code:`FlowProposal` class.

Custom proposals
----------------

Custom proposals should inherit from
:py:class:`nessai.proposal.flowproposal.FlowProposal` and made available using
the ``nessai.proposals`` entry point.
Once a custom proposal is installed, it can be used by specifying the
:code:`flow_proposal_class` keyword argument when calling either
:py:class:`~nessai.samplers.nestedsampler.NestedSampler` or
:py:class:`~nessai.flowsampler.FlowSampler`.



Known plugins
-------------

* ``nessai-gw`` (`Homepage <https://github.com/mj-will/nessai-gw>`_) - provides gravitational-wave specific proposals and reparameterisations.
