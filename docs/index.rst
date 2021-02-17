.. nessai documentation master file, created by
   sphinx-quickstart on Wed Sep 16 13:23:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nessai's documentation!
========================================

.. note::
    This page is still under construction. Some links may not work and some sections are missing.

Nessai
======

``nessai`` (/ˈnɛsi/): Nested Sampling with Aritificial Intelligence

``nessai`` is a nested sampling algorithm for Bayesian Inference that incorporates normalisings flows. It is designed for applications where the Bayesian likelihood is computationally expensive.

The code is available at: https://github.com/mj-will/nessai.

.. automodule:: nessai
   :members:

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   running-the-sampler
   sampler-configuration
   reparameterisations
   normalising-flows-configuration
   further-details

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   gaussian-example
   bilby-example


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
