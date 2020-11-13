============
Installation
============

Some of the dependencies are not available for Conda, so installation is slightly different for conda vs pip.

NOTE: this packages requieres python >= 3.8


Installing nessai
=================

PyTorch
-------

Nessai requires PyTorch will can be installed using `pip` or `conda`. It's recommend to first install the correct version on PyTorch for your system and then proceed with the usual installation. See the Pytorch homepage for instructions: https://pytorch.org/.

NOTE: GPU support is currently supported but not recommended as for the majority of use-cases training on the GPU is slower than CPU

Nessai
------

Nessai can be installed using `pip`

.. code-block:: console

    $ pip install nessai

This will install the dependencies needed but the version of `torch` will no necessarily be the correct version for your system unless you followed the previous step.

Nessai for development
----------------------

To install nessai for development purposes see the contribution guidelines.



Using nessai with bilby
=======================

This pacakge requieres a fork of Bilby that includes the sampler, it can be installed by running (this requires pip):

```
pip install git+https://git.ligo.org/michael.williams/bilby.git@add-nessai-sampler#egg=bilby
```
