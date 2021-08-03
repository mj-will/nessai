###################
Reparameterisations
###################

In ``nessai`` three spaces are defined:

- the sampling space :math:`\mathcal{X}`,
- the reparameterised sampling space :math:`\mathcal{X}'` (or prime space),
- the latent space :math:`\mathcal{Z}`.

The key to efficient sampling with ``nessai`` is to reparameterise the sampling space such that the prime space is simpler for normalising flow to learn.

There are two main methods for configuring reparameterisations in ``nessai``. The first is simpler but more limited and the second is more complex but allows for greater control of the specific reparameterisations used.

************************************
Method 1: general reparameterisation
************************************

This method is limited to the following types of reparameterisation:

- rescaling to a fixed range,
- applying inversion along a bound.

These can be applied on parameter by parameter basis using keyword arguments. The relevant arguments:

- :code:`rescale_parameters`: This is either a :code:`bool` or :code:`list` of parameters. If :code:`True`, then all of the parameters in the model are rescaled. If a list is provided only those parameters are rescaled. It's generally recommend to rescale parameters to a common range to improve the efficiency of training. The inputs are rescaled to the range defined by :code:`rescale_bounds`.
- :code:`rescale_bounds`: A :code:`list` of length two containing the upper and lower bounds of the range for rescaling, the default value is [-1, 1].
- :code:`update_bounds`: A boolean, if :code:`True` the minimum and maximum values of parameters are update each time the normalising flow is trained. If :code:`False` the parameters are always rescaled using the initial prior bounds. It's recommended to enable this option.
- :code:`boundary_inversion`: This is either a :code:`bool` or :code:`list` and determines if boundary inversion is applied to any of the parameters. If :code:`True` it is applied to all parameters and if a list is specified it is only applied to those parameters. This forces the samples for the parameters and one of the bounds to be mirrored along that bound. The bounds is chosen based on the density of samples. It's not recommended to enable this setting unless there are parameters for which the posterior distributions will consistently rail against the prior bounds. NOTE: this forces :code:`update_bounds=True`.
- :code:`detect_edges`: This setting is only relevant when using :code:`boundary_inversion` and it allows the sampler to detect hard bounds in the samples (or lack thereof)


**************************************
Method 2: specific reparameterisations
**************************************

This methods allows for use of the all the reparameterisations included in ``nessai``. However it requires parameters to be configured individually. All the reparameterisations are configure using the keyword argument `reparameterisations` which is an instance of :code:`dict`.

When using this method reparameterisations are added to the proposal method based on the dictionary. Each entry in the reparameterisations dictionary is interpreted and add to the combined reparameterisation that is applied to all parameters.
These the following key-value pairs are understood:

- **Parameter & Reparameterisation**: the key is the parameter to which the reparameterisation is applied and the reparameterisation to apply. For example :code:`reparameterisations={'x': 'default'}`, this tells the sampler to use the default reparameterisation for x, which rescales to [-1, 1].

- **Parameter & Kwargs**: the key is the same as in the previous case but instead of the name of the reparameterisation, a dictionary with the configuration is specified. For example: :code:`reparamterisations={'x': {'reparameterisation': 'default', 'rescale_bounds': [0, 1]}}`, this tells the sampler to use the default reparameterisation but with a specific keyword argument :code:`rescale_bounds`. The resulting reparameterisation will rescale to [0, 1] instead of [-1, 1].

- **Reparameterisation & Kwargs**: here the key is the name of reparameterisation and the kwargs are the used to configure the reparameterisation. This dictionary MUST contain the name of the parameter(s) to which the reparameterisation is applied. For example: :code:`reparameterisation={'default': {'parameters': ['x'], 'rescale_bounds':[0, 1]}`. This applied the default reparameterisation to x but with rescaling to [0, 1]. This method also supports specifying multiple parameters for reparameterisations that support it, for example: :code:`reparameterisation={'default': 'parameters': ['x', 'y']}`. This is necessary for reparameterisations that are applied to groups of parameters, such as the pairs of angles.


See the `examples directory <https://github.com/mj-will/nessai/tree/master/examples>`_ for an example of using this method of defining the reparmeterisations.

.. note::
    Missing section on use of x prime prior


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
- ``'none'``: No reparameterisation is applied.

For details on each of these reparameterisations see :py:mod:`~nessai.reparameterisations`.
