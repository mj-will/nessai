===============================
Normalising flows configuration
===============================


Nessai uses the implementation of normalising flow avaiable in ``nflows`` but with some minor changes to make the iterface more general.

The normalising flow is configured using the the keyword argument ``flow_config`` when calling ``FlowSampler``. This is a dictionary which contains the configuration for training and the flow itself which is another dictionary ``model_config``.

The hyper-parameters accepted in ``model_config`` are:

- ``n_blocks``: number transforms to use
- ``n_layers``: number of layers to use the neural network in each transform
- ``n_neurons``: number of neurons per layer in the neural network
- ``ftype``: type of normalising flow to use, see :ref:``Included normalising flows``
- ``device_tag``: device on which to train the normalising flow, defaults to ``'cpu'``
- ``kwargs``: keyword arguments parsed to the flow class used, e.g. ``linear_transform`` or ``batch_norm_between_layers``


Included normalising flows
--------------------------

Nessai includes three different normalising flow out-of-the-box and can be specified using ``ftype``, these
are:

- RealNVP (``'realnvp'``)
- MaskedAutoregressiveFlows (``'maf'``)
- Neural Spline Flows (``'nsf'``)


Using other normalising flows
-----------------------------

Other normalising flows can be implemented by the user and used with nessai
by specifying the ``flow`` parameter in the ``model_config`` input dictionary as an object that inherits from
``nessai.flows.base.BaseFlow`` and redefines all of the methods. The object will initialised within the sampler using ``nessai.flows.utils.setup_model`` and ``model_config``.

Alternatively flows can implemented using same approachs as ``nflows`` using ``nessai.flows.base.NFlow`` where a ``transform`` and ``distribution`` are specified. The ``__init__`` method must accept the same arguments as described for ``BaseFlow``. For an example of how to use this method see the implementations of either RealNVP or Neural Spline Flows.
