===============================
Normalising flows configuration
===============================


``nessai`` uses the implementation of normalising flow available in ``glasflow`` which is based on ``nflows``. The exact interface in ``nessai`` is slightly different to allow for some extra functionality.

The normalising flow is configured using the the keyword argument ``flow_config`` when calling :py:class:`~nessai.flowsampler.FlowSampler`. This is a dictionary which contains the configuration for training and the flow itself which is another dictionary ``model_config``.

The hyper-parameters accepted in ``model_config`` are:

- ``n_blocks``: number transforms to use
- ``n_layers``: number of layers to use the neural network in each transform
- ``n_neurons``: number of neurons per layer in the neural network
- ``ftype``: type of normalising flow to use, see :ref:`included normalising flows<Included normalising flows>`
- ``device_tag``: device on which to train the normalising flow, defaults to ``'cpu'``
- ``kwargs``: keyword arguments parsed to the flow class used, e.g. ``linear_transform`` or ``batch_norm_between_layers``

The remaining items in :code:`flow_config` control the training and these are:

- :code:`lr`: the learning rate used to train the model, default is 0.001
- :code:`batch_size`: the batch size to use for training
- :code:`val_size`: the fraction of data to use for validation
- :code:`max_epochs`: the maximum number of epochs to train for
- :code:`patience`: the number of iterations with no improvement in the validation loss to wait before stopping training early
- :code:`annealing`: enable learning rate annealing
- :code:`clip_grad_norm`: clipping used for the gradient
- :code:`noise_scale`: scale of the Gaussian noise added to the data. Proposed in Moss 2019.

The default settings are:

.. code:: python

    default = dict(
        lr=0.001,
        batch_size=100,
        val_size=0.1,
        max_epochs=500,
        patience=20,
        annealing=False,
        clip_grad_norm=5,
        noise_scale=0.0
    )

Example configuration
=====================

Here's an example of what a configuration could look like:

.. code:: python

    flow_config = dict(
        lr=3e-3,
        batch_size=1000,
        max_epochs=500,
        patience=20,
        model_config=dict(
            n_blocks=4,
            n_layers=2,
            n_neurons=16,
            kwargs=dict(linear_transform='lu')
        )
    )


This could then be parsed directly to :py:class:`~nessai.flowsampler.FlowSampler`.


Included normalising flows
==========================

``nessai`` includes three different normalising flow out-of-the-box and can be specified using ``ftype``, these
are:

- RealNVP (``'realnvp'``)
- MaskedAutoregressiveFlows (``'maf'``)
- Neural Spline Flows (``'nsf'``)


Using other normalising flows
=============================

Other normalising flows can be implemented by the user and used with nessai by specifying the :code:`flow` parameter in the :code:`model_config` input dictionary as an object that inherits from :py:class:`nessai.flows.base.BaseFlow` and redefines all of the methods. The object will initialised within the sampler using :py:func:`nessai.flows.utils.setup_model` and :code:`model_config`.

Alternatively flows can implemented using same approach as ``glasflow.nflows`` using :py:class:`nessai.flows.base.NFlow` where a ``transform`` and ``distribution`` are specified. The ``__init__`` method must accept the same arguments as described for :py:class:`~nessai.flows.base.BaseFlow`. For an example of how to use this method see the implementations of either RealNVP or Neural Spline Flows.


Using nflows instead of glasflow
================================

``nessai`` migrated to using ``glasflow`` since this removes the dependency on ``nflows``, however it is still possible to use a locally installed version of ``nflows``. ``glasflow`` includes a fork of ``nflows`` as a submodule and this can be replaced with a local install by setting an environment variable:

.. code:: bash

    export GLASFLOW_USE_NFLOWS=True


``nesssai`` will still import ``glasflow`` but ``glasflow.nflows`` will point to the local install of ``nflows``.
