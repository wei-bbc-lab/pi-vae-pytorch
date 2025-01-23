import torch
from torch import nn, Tensor


class ELBOLoss(nn.Module):
    """ 
    pi-VAE Loss function

    Parameters
    ----------
    observation_model (str) - poisson or gaussian. Default: poisson
    device (torch.device) - object representing the device on which a Tensor will be allocated. Default: None 
    
    Inputs
    ------
    x (Tensor) - observed x. Size([n_samples, x_dim])
    firing_rate (Tensor) - decoded firing rates. Size([n_samples, x_dim])
    lambda_mean (Tensor) - means from label prior p(z|u). Size([n_samples, z_dim])
    lambda_log_variance (Tensor) - log of variances from label prior p(z|u). Size([n_samples, z_dim])
    posterior_mean (Tensor) - means from posterior q(z|x,u)~q(z|x)p(z|u). Size([n_samples. z_dim])
    posterior_log_variance (Tensor) - log of variances from posterior q(z|x,u)~q(z|x)p(z|u). Size([n_samples, z_dim])
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

    def __init__(
        self,
        observation_model: str = 'poisson',
        device: torch.device = None
        ) -> None:
        super(ELBOLoss, self).__init__()

        if observation_model == 'gaussian':
           self.ones = torch.ones(size=(1, 1), device=device)
        elif observation_model != 'poisson' and observation_model != 'gaussian':
            raise ValueError(f"Invalid observation_model: {observation_model}")
        
        self.observation_model = observation_model

    def forward(
        self,
        x: Tensor,
        firing_rate: Tensor,
        lambda_mean: Tensor, 
        lambda_log_variance: Tensor,
        posterior_mean: Tensor, 
        posterior_log_variance: Tensor,
        observation_noise_model: nn.Module = None
        ) -> Tensor:

        if self.observation_model == 'poisson':
            observation_log_liklihood = torch.sum(firing_rate - x * torch.log(firing_rate), dim=-1)
        else:
            observation_log_variance = observation_noise_model(self.ones)
            observation_log_liklihood = torch.sum(torch.square(firing_rate - x) / (2 * torch.exp(observation_log_variance)) + (observation_log_variance / 2), dim=-1)

        kl_loss = 1 + posterior_log_variance - lambda_log_variance - ((torch.square(posterior_mean - lambda_mean) + torch.exp(posterior_log_variance)) / torch.exp(lambda_log_variance))
        kl_loss = 0.5 * torch.sum(kl_loss, dim=-1)

        return torch.mean(observation_log_liklihood - kl_loss)
