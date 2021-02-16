============
Installation
============

The prefered installation method is via ``pip`` in the virtual environment of your choice.

.. note::
    Nessai requires Python 3.8 or greater.


Installing nessai
=================

PyTorch
-------

Nessai requires PyTorch will can be installed using ``pip`` or ``conda``. It's recommend to first install the correct version on PyTorch for your system and then proceed with the usual installation. See the PyTorch homepage for instructions: https://pytorch.org/.

.. note::
    Nessai includes GPU support but it is not recommended as for the majority of use-cases the current implmentation runs slower on the GPU than CPU.

Nessai
------

Nessai can be installed using ``pip``

.. code-block:: console

    $ pip install nessai

This will install the dependencies needed but the version of PyTorch will not necessarily be the correct version for your system unless you followed the previous step.

Nessai for development
----------------------

To install nessai for development purposes see the `contribution guidelines <https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md>`_.


Using nessai with bilby
=======================

This pacakge requiers a fork of Bilby that includes the sampler. This will hopefully change in near future but for now it can be installed by running:

.. code-block:: console

    $ pip install git+https://git.ligo.org/michael.williams/bilby.git@add-nessai-sampler#egg=bilby
    
This branch fork is based on version 1.0.3 of bilby.

