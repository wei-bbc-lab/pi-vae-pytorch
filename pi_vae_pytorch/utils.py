from typing import List

import torch
from torch import nn, Tensor


def build_mlp_layers(
    n_hidden_layers: int,
    hidden_layer_dims: List[int],
    activation: nn.Module
    ) -> nn.Sequential:
    """
    Helper method to construct a Multilayer Perceptron (MLP). 

    Parameters
    ----------
    n_hidden_layers (int) - the number of hidden in the model
    hidden_layer_dim (List[int]) - a list of integers specifying the input/output dimensions of each layer
    activation (nn.Module) - the activation function to apply after each hidden layer

    Returns
    -------
    nn.Sequential(*layers) - a Sequential container of the specified layers

    Notes
    -----
    A nn.Identity activation is applied after the final layer.
    """

    layers = []
    act = activation

    # Build layers
    for i in range(n_hidden_layers+1):
        # Output layer
        if i == n_hidden_layers:
            act = nn.Identity
        
        layers.append(nn.Linear(hidden_layer_dims[2*i], hidden_layer_dims[2*i+1]))
        layers.append(act())

    return nn.Sequential(*layers)
