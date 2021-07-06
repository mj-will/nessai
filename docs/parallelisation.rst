###########################
Parallelisation with nessai
###########################

One benefit of the proposal method used in ``nessai`` is it allows for simple parallelisation of the likelihood evaluation since new live points are drawn in batches and then stored. The likelihood can therefore be precomputed and stored later use.

************************
Enabling parallelisation
************************

There are two keyword arguments that must be set to enable parallelisation:

- :code:`n_pool`: The number of threads to use for evaluating the likelihood
- :code:`max_threads`: The maximum number of threads to use, this should be at least 1 larger than :code:`n_pool`. Extra threads are allocated to PyTorch's CPU parallelisation.

.. note::
    If running ``nessai`` via a job scheduler such as HTCondor, remember to set the number of requested threads accordingly. This should match :code:`max_threads`.


*************
Example usage
*************

.. literalinclude:: ../examples/parallelisation_example.py
    :language: python

********
Also see
********

- :py:func:`nessai.utils.configure_threads`
- :py:meth:`nessai.proposal.base.Proposal.configure_pool`
- :py:meth:`nessai.proposal.base.Proposal.close_pool`
