Sampling with discrete parameters
=================================

The normalising flows in ``nessai`` do not support discrete parameters.
As such, any discrete parameters in a model must be handled differently to
other parameters.

.. note::

    The importance nested sampler does not currently support sampling with
    discrete parameters.

==============
Dequantisation
==============

Dequantisation is a technique used to convert discrete data into continuous data.
It is designed for use with integer values and works by adding random noise
on :math:`[0, 1)` to the data:

.. math::

    x_{dequantise} = x + a \quad \text{where} \quad a \sim U[0, 1).


In ``nessai``, dequantisation is available via a reparameterisation called
``dequantise``. See below for an example of how it is used.


============================
Example using dequantisation
============================


.. literalinclude:: ../examples/discrete_parameter.py
    :language: python
