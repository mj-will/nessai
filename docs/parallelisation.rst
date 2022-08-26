###########################
Parallelisation with nessai
###########################

One benefit of the proposal method used in ``nessai`` is it allows for simple parallelisation of the likelihood evaluation since new live points are drawn in batches and then stored. The likelihood can therefore be precomputed and stored later use.

************************
Enabling parallelisation
************************

Likelihood parallelisation can be enabled in ``nessai`` by setting the keyword argument :code:`n_pool` when calling :code:`FlowSampler`. This determines the size of the multiprocessing pool to use for evaluating the likelihood.

.. note::
    If running ``nessai`` via a job scheduler such as HTCondor, remember to set the number of requested CPUs accordingly.


*****************
Specifying a pool
*****************

Alternatively, ``nessai`` can use a user-defined pool. This is specified by setting the :code:`pool` argument in :code:`NestedSampler` or :code:`FlowSampler`. Some variables must be initialised when creating the pool, this is done using :py:func:`~nessai.utils.multiprocessing.initialise_pool_variables`:

.. code-block:: python

    from multiprocessing import Pool
    from nessai.utils.multiprocessing import initialise_pool_variables

    model = GaussianModel()
    pool = Pool(
        processes=2,
        initializer=initialise_pool_variables,
        initargs=(model,),
    )

:code:`pool` can then passed to the :code:`pool` keyword argument when setting up the sampler.

-------------
Using ``ray``
-------------

``ray`` includes a `distributed multiprocessing pool <https://docs.ray.io/en/latest/ray-more-libs/multiprocessing.html>`_ that can also be used with ``nessai``. Simply import :code:`ray.util.multiprocessing.Pool` instead of the standard pool and initialise using the method described above.


-----------
Other pools
-----------

When a pool object is passed to :code:`nessai` it tries to determine how many processes the pool contains and (if the likelihood is vectorised) uses this information to determine the chunk size when evaluating the likelihood. If it can not determine this, then likelihood vectorisation will be disabled. This can be avoided by specifying :code:`n_pool` and :code:`max_threads` when initialising the sampler.


*************
Example usage
*************

.. literalinclude:: ../examples/parallelisation_example.py
    :language: python

********
See also
********

- :py:func:`nessai.utils.threading.configure_threads`
- :py:func:`nessai.utils.multiprocessing.initialise_pool_variables`
- :py:meth:`nessai.model.Model.configure_pool`
- :py:meth:`nessai.model.Model.close_pool`
