# NesSAI: Nested Sampling with Aritfical Intelligence

Please note that this package is still in early development and is not publicly available.

## Installation

NOTE: this packages requieres python >= 3.8

Nessai current is not available to install from `pypa`, so for now installation requires installing from source.

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

```
pip install git+https://git.ligo.org/michael.williams/bilby.git@flowproposal-sampler#egg=bilby
```


## Contributing

Please see the guidelines our guidelines [here]()https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md.
