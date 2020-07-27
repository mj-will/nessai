# FlowProposal

Please note that this package is still in early development and is not publicly available.

## Installation

Some of the dependencies are not available for Conda, so installation is slightly different for conda vs pip.

NOTE: this packages requieres python >= 3.8

### PyTorch

Many of the dependencies are shared with Pytorch, so first install the version of Pytorch in your Python environment. See the Pytorch homepage for instructions: https://pytorch.org/. 

NOTE: GPU support is currently disabled but should be added in the future.

### Installing FlowProposal

First clone the repository (cloning with SSH requires having the host for Gilsay configured):

```
git clone https://gilsay.physics.gla.ac.uk/gitlab/michael.williams/flowproposal.git
```

Then to install the package (in the Python environemt you want to use):

```
cd flowproposal
python setup.py install
```
Any missing dependencies should be installed.

### Adding Bilby

This pacakge requieres a fork of Bilby that includes the sampler, it can be installed by running (this requires pip):

```
pip install git+https://git.ligo.org/michael.williams/bilby.git@flowproposal-sampler#egg=bilby
```
