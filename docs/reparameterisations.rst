===================
Reparameterisations
===================

In ``nessai`` three spaces are defined:

- the sampling space :math:`\mathcal{X}`,
- the reparameterised sampling space :math:`\mathcal{X}'` (or prime space),
- the latent space :math:`\mathcal{Z}`.

The key to efficient sampling with ``nessai`` is to reparameterise the sampling space such that the prime space is simpler for normalising flow to learn.

The reparameterisations are controlled via three keyword arguments:

- :code:`reparameterisations: dict|str|None`: either a dictionary specifying various reparameterisations, a string specifying a single reparameterisation for all parameters or `None` to use the defaults/fallback. See :ref:`configuring reparameterisations<Configuring reparameterisations>` for more details.
- :code:`fallback_reparameterisations: None|str`: reparameterisation use if a reparameterisation has not been specified for a parameter. If `None`, no reparameterisation is used.
- :code:`use_default_reparameterisations: bool|None`: if :code:`True` any default reparameterisations included in the proposal class, e.g. those in :code:`GWFlowProposal` will be used for any parameters not specified in :code:`reparameterisations`. If :code:`False`, these defaults will be ignored. If :code:`None`, the behaviour will depende on the proposal class.
- :code:`reverse_reparameterisations: bool`: if :code:`True`, the order of the reparmeterisations will be reversed.


Configuring reparameterisations
===============================

The :code:`reparameterisations` keyword allows for fine-grained configuration of the reparameterisations using a dictionary.
Each entry in the reparameterisations dictionary is interpreted and add to the combined reparameterisation that is applied to all parameters.
The following key-value pairs are understood:

- **Parameter & Reparameterisation**: the key is the parameter to which the reparameterisation is applied and the reparameterisation to apply. For example :code:`reparameterisations={'x': 'default'}`, this tells the sampler to use the default reparameterisation for x, which rescales to [-1, 1].

- **Parameter & Kwargs**: the key is the same as in the previous case but instead of the name of the reparameterisation, a dictionary with the configuration is specified. For example: :code:`reparamterisations={'x': {'reparameterisation': 'default', 'rescale_bounds': [0, 1]}}`, this tells the sampler to use the default reparameterisation but with a specific keyword argument :code:`rescale_bounds`. The resulting reparameterisation will rescale to [0, 1] instead of [-1, 1].

- **Reparameterisation & Kwargs**: here the key is the name of reparameterisation and the kwargs are the used to configure the reparameterisation. This dictionary MUST contain the name of the parameter(s) to which the reparameterisation is applied. For example: :code:`reparameterisation={'default': {'parameters': ['x'], 'rescale_bounds':[0, 1]}`. This applied the default reparameterisation to x but with rescaling to [0, 1]. This method also supports specifying multiple parameters for reparameterisations that support it, for example: :code:`reparameterisation={'default': 'parameters': ['x', 'y']}`. This is necessary for reparameterisations that are applied to groups of parameters, such as the pairs of angles. This methods also supports regex for configuring multiple parameters.


See the `examples directory <https://github.com/mj-will/nessai/tree/master/examples>`_ for an example of using this method of defining the reparmeterisations.


Available reparameterisations
=============================

There are a number of pre-configured reparameterisations included in ``nessai``:

- ``default``: Default rescaling which rescales to [-1, 1] and has :code`update_bounds=True`
- ``rescaletobounds``: Alias for default
- ``inversion``:  Default rescaling but with inversion and :code:`detect_edges` enabled for the parameter, uses :code:`split` inversion.
- ``inversion-duplicate``: Same as :code:`inversion` but uses :code:`duplicate` inversion
- ``offset``: Equivalent to :code:`default` but includes an offset before rescaling. This is usual when dealing with parameters which have small prior ranges but are offset from zero by some large constant. For example time, which is typically of order :math:`10^{9}`
- ``logit``: Parameters are rescaled to [0, 1] and a logit is applied. A sigmoid is used for the inverse. **Note:** :code:`update_bounds` is disabled by default.
- ``angle``: Reparameterisation for angles. This reparameterisation introduces an auxiliary radial parameter and converts the angle to Cartesian coordinates.
- ``angle-pi``: Same as :code:`angle` but specifically for angles defined on [0, :math:`\pi`]
- ``angle-sine``: Same as :code:`angle` but for angles with sine priors
- ``angle-2pi``: Same as :code:`angle-pi` but for angles defined on [0, :math:`2\pi`]
- ``angle-pair``: Reparameterisation for pairs of angles, see :py:class:`nessai.reparameterisations.AnglePair` for details.
- ``dequantise``: Reparameterisation for discrete parameters for that adds random noise to each integer values.
- ``'none'``: No reparameterisation is applied.

For details on each of these reparameterisations see :py:mod:`~nessai.reparameterisations`.
