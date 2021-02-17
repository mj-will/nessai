# Nessai: Nested Sampling with Artificial Intelligence

``nessai`` (/ˈnɛsi/): Nested Sampling with Aritificial Intelligence

``nessai`` is a nested sampling algorithm for Bayesian Inference that incorporates normalisings flows. It is designed for applications where the Bayesian likelihood is computationally expensive.

## Installation

NOTE: this packages requieres python >= 3.8

``nessai`` currently is not available to install from `PyPI`, so for now installation requires installing from source.

First clone the repo:

```console
$ git clone git@github.com:mj-will/nessai.git
```

The to install the package:

```console
$ cd nessai
$ pip install .
```

### PyTorch

By default the CPU only version of PyTroch will installed, to install a different version with CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.

### Adding Bilby

This package requieres a fork of Bilby that includes the sampler, it can be installed by running (this requires pip):

```console
pip install git+https://git.ligo.org/michael.williams/bilby.git@add-nessai-sampler#egg=bilby
```

## Documentation

Documenation will be added to the ReadTheDocs shortly.

## Contributing

Please see the guidelines guidelines [here](https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md).


## Acknowledgements

The core nested sampling code, model design and code for computing the posterior in ``nessai`` was based on [`cpnest`](https://github.com/johnveitch/cpnest) with permission from the authors.

The normalising flows implemented in ``nessai`` are all either directly imported from [`nflows`](https://github.com/bayesiains/nflows/tree/master/nflows) or heavily based on it.

Other code snippets that draw on existing code reference the source in their corresponding doc-strings.
