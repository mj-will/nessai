============
Installation
============

.. note::

    It is recommended to install PyTorch before installing ``nessai``.
    See the `PyTorch documentation <https://pytorch.org/>`_ for how to install the correct version.

.. tabs::

     .. code-tab:: console conda

        conda install -c conda-forge -c pytorch nessai

     .. code-tab:: console pip

        pip install nessai


This will install the dependencies needed but the version of PyTorch will not
necessarily be the correct version for your system.


.. _nessai-bilby:

Using nessai with bilby
=======================


As of ``bilby`` version 2.3.0, the recommended way to use ``nessai`` is via
the ``nessai-bilby`` sampler plugin.
This can be installed via either ``conda`` or ``pip`` and provides the most
up-to-date interface for ``nessai``.
This includes support for the importance nested sampler (``inessai``).

.. tabs::

     .. code-tab:: console conda

        conda install -c conda-forge nessai-bilby

     .. code-tab:: console pip

        pip install nessai-bilby


For more details about the plugin, see the ``nessai-bilby``
`documentation <https://github.com/bilby-dev/nessai-bilby>`_


For older versions of ``bilby`` (<2.3.0 and >1.1.0), no additional packages
are required to use ``nessai`` but some versions may be incompatible.

For examples of how to run ``nessai`` via ``bilby`` see the
`examples directory <https://github.com/mj-will/nessai/tree/main/examples>`_.


Developing for nessai
=====================

To install ``nessai`` for development purposes see the
`contribution guidelines <https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md>`_.
