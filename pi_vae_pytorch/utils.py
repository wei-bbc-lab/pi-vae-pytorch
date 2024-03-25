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

def compute_loss(
    x: Tensor,
    firing_rate: Tensor,
    lambda_mean: Tensor, 
    lambda_log_variance: Tensor,
    posterior_mean: Tensor, 
    posterior_log_variance: Tensor,
    observation_model: str,
    observation_noise_model: nn.Module = None
    ) -> Tensor:
    """ 
    pi-VAE Loss function

    Parameters
    ----------
    x (Tensor) - observed x. Size([n_samples, x_dim])
    firing_rate (Tensor) - decoded firing rates. Size([n_samples, x_dim])
    lambda_mean (Tensor) - means from label prior p(z|u). Size([n_samples, z_dim])
    lambda_log_variance (Tensor) - log of variances from label prior p(z|u). Size([n_samples, z_dim])
    posterior_mean (Tensor) - means from posterior q(z|x,u)~q(z|x)p(z|u). Size([n_samples. z_dim])
    posterior_log_variance (Tensor) - log of variances from posterior q(z|x,u)~q(z|x)p(z|u). Size([n_samples, z_dim])
    observation_model (str) - poisson or gaussian
    observation_noise_model (nn.Module) - if gaussian observation model, set the observation noise level as different real numbers. Default: None 
    
    Returns
    -------
    The total loss (Tensor)

    Notes
    -----
    observation_noise_model (nn.Module) - PiVAE uses an nn.Linear(1, x_dim, bias=False) Module

    min -log p(x|z) + E_q log(q(z))-log(p(z|u))
    cross entropy
    q (mean1, var1) p (mean2, var2)
    E_q log(q(z))-log(p(z|u)) = -0.5*(1-log(var2/var1) - (var1+(mean2-mean1)^2)/var2)
    E_q(z|x,u) log(q(z|x,u))-log(p(z|u)) = -0.5*(log(2*pi*var2) + (var1+(mean2-mean1)^2)/var2)
    p(z) = q(z|x) = N(f(x), g(x)) parametrized by nn
    """

    if observation_model == "poisson":
        observation_log_liklihood = torch.sum(firing_rate - x * torch.log(firing_rate), -1)
    elif observation_model == "gaussian":
        observation_log_variance = observation_noise_model(torch.ones((1, 1)))
        observation_log_liklihood = torch.sum(torch.square(firing_rate - x) / (2 * torch.exp(observation_log_variance)) + (observation_log_variance / 2), -1)
    else:
        raise ValueError(f"Invalid observation model: {observation_model}")

    loss = 1 + posterior_log_variance - lambda_log_variance - ((torch.square(posterior_mean - lambda_mean) + torch.exp(posterior_log_variance)) / torch.exp(lambda_log_variance))
    kl_loss = 0.5 * torch.sum(loss, dim=-1)

    return torch.mean(observation_log_liklihood - kl_loss)

def compute_posterior(
    z_mean: Tensor, 
    z_log_variance: Tensor, 
    lambda_mean: Tensor, 
    lambda_log_variance: Tensor,
    ) -> Tensor:
    """
    Compute the full posterior of q(z|x,u)~q(z|x)p(z|u) as a product of Gaussians.

    Parameters
    ----------
    z_mean (Tensor) - means of encoded distribution q(z|x). Size([n_samples, z_dim])
    z_log_variance (Tensor) - log of varinces of encoded distribution q(z|x). Size([n_samples, z_dim])
    lambda_mean (Tensor) - means of label prior distribution p(z|u). Size([n_samples, z_dim])
    lambda_log_variance (Tensor) - log of variances of label prior distribution p(z|u). Size([n_samples, z_dim])

    Returns
    -------
    posterior_mean: approximate posterior means of distribution q(z|x,u). Size([n_samples, z_dim])
    posterior_log_variance: approximate posterior log of variances of distribution q(z|x,u). Size([n_samples, z_dim])
    """

    variance_difference = z_log_variance - lambda_log_variance
    posterior_mean = (z_mean / (1 + torch.exp(variance_difference))) + (lambda_mean / (1 + torch.exp(torch.neg(variance_difference))))
    posterior_log_variance = z_log_variance + lambda_log_variance - torch.log(torch.exp(z_log_variance) + torch.exp(lambda_log_variance))

    return posterior_mean, posterior_log_variance

def generate_latent_z(
    mean: Tensor,
    log_variance: Tensor
    ) -> Tensor:
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.

    Parameters
    -----------
    mean (Tensor) - means of each z sample. Size([n_samples, z_dim])
    log_variance (Tensor) - log of variances of each z sample. Size([n_samples, z_dim])

    Returns
    ------
    Samples of latent z. Size([n_samples, z_dim])
    """

    return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
