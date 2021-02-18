# Nessai: Nested Sampling with Artificial Intelligence

``nessai`` (/ˈnɛsi/): Nested Sampling with Aritificial Intelligence

``nessai`` is a nested sampling algorithm for Bayesian Inference that incorporates normalisings flows. It is designed for applications where the Bayesian likelihood is computationally expensive.

## Installation

``nessai`` can be installed using ``pip``:

```console
$ pip install nessai
```

Installing via ``conda`` is not currently supported.

### PyTorch

By default the version of PyTroch will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.

### Adding Bilby

This package requieres a fork of Bilby that includes the sampler, it can be installed by running (this requires pip):

```console
$ pip install git+https://git.ligo.org/michael.williams/bilby.git@add-nessai-sampler#egg=bilby
```

## Documentation

Documenation is available at: [nessai.readthedocs.io](https://nessai.readthedocs.io/)

## Contributing

Please see the guidelines [here](https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md).


## Acknowledgements

The core nested sampling code, model design and code for computing the posterior in ``nessai`` was based on [`cpnest`](https://github.com/johnveitch/cpnest) with permission from the authors.

The normalising flows implemented in ``nessai`` are all either directly imported from [`nflows`](https://github.com/bayesiains/nflows/tree/master/nflows) or heavily based on it.

Other code snippets that draw on existing code reference the source in their corresponding doc-strings.
