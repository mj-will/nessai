# -*- coding: utf-8 -*-
"""
Neural networks for use in flows.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """A standard multi-layer perceptron.

    Based on the implementation in nflows and modified to include dropout and
    conditional inputs.

    Parameters
    ----------
    in_shape : tuple
        Input shape.
    out_shape : tuple
        Output shape.
    hidden_sizes : List[int]
        Number of neurons in the hidden layers.
    activation : Callable
        Activation function
    activate_output : Union[bool, Callable]
        Whether to activate the output layer. If a bool is specified the same
        activation function is used. If a callable inputs is specified, it
        will be used for the activation.
    dropout_probability : float
        Amount of dropout to apply after the hidden layers.
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
        dropout_probability=0.0,
    ):
        super().__init__()
        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(
                    hidden_sizes[:-1], hidden_sizes[1:]
                )
            ]
        )
        self._dropout_layers = nn.ModuleList(
            nn.Dropout(dropout_probability)
            for _ in range(len(self._hidden_layers))
        )
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

        if activate_output:
            self._activate_output = True
            if activate_output is True:
                self._output_activation = self._activation
            elif callable(activate_output):
                self._output_activation = activate_output
            else:
                raise TypeError(
                    "activate_output must be a boolean or a callable"
                )
        else:
            self._activate_output = False

    def forward(self, inputs, context=None):
        """Forward method that allows for kwargs such as context.

        Parameters
        ----------
        inputs : :obj:`torch.tensor`
            Inputs to the MLP
        context : None
            Conditional inputs, must be None. Only implemented to the
            function is compatible with other methods.

        Raises
        ------
        ValueError
            If the context is not None.
        ValueError
            If the input shape is incorrect.
        """
        if context is not None:
            raise ValueError("MLP with conditional inputs is not implemented.")
        if inputs.shape[1:] != self._in_shape:
            raise ValueError(
                "Expected inputs of shape {}, got {}.".format(
                    self._in_shape, inputs.shape[1:]
                )
            )

        inputs = inputs.reshape(-1, np.prod(self._in_shape))
        outputs = self._input_layer(inputs)
        outputs = self._activation(outputs)

        for hidden_layer, dropout in zip(
            self._hidden_layers, self._dropout_layers
        ):
            outputs = hidden_layer(outputs)
            outputs = self._activation(outputs)
            outputs = dropout(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._output_activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)

        return outputs
