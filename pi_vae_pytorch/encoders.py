from typing import Tuple

import torch
from torch import nn, Tensor

from pi_vae_pytorch.layers import MLP


class MLPEncoder(nn.Module):
    """
    Defines mean and log of variance of q(z|x).

    Parameters
    ----------
    x_dim (int) - observed x dimension
    z_dim (int) - latent z dimension
    n_hidden_layers (int) - number of MLP hidden layers. Default: 2
    hidden_layer_dim (int) - dimension of each MLP hidden layer. Default: 120
    activation (nn.Module) - activation function applied to each MLP hidden layer. Default: nn.Tanh
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        n_hidden_layers: int = 2,
        hidden_layer_dim: int = 120,
        activation: nn.Module = nn.Tanh
        ) -> None:
        super().__init__()

        self.net = MLP(
            in_features=x_dim,
            out_features=z_dim*2,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_features=hidden_layer_dim,
            activation=activation
        )
    
    def forward(
        self,
        x: Tensor
        ) -> Tuple[Tensor, Tensor]:
        """
        Maps observed x to mean and log of variance of q(z|x).
        """

        q_z = self.net(x)
        # phi_mean, phi_log_variance
        return torch.chunk(q_z, 2, -1)
