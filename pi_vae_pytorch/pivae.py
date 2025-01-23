from typing import Union, Optional

import torch
from torch import nn

from pi_vae_pytorch.decoders import GINFlowDecoder
from pi_vae_pytorch.encoders import MLPEncoder
from pi_vae_pytorch.layers import ZPriorContinuous, ZPriorDiscrete


class PiVAE(nn.Module):
    """
    The pi-VAE model which constructs a latent variable model of neural activity while 
    simultaneously modeling the relation between the latent and task variables.

    Parameters
    ----------
    x_dim (int) - observed x dimension
    u_dim (int) - 
    z_dim (int) - latent z dimension
    discrete_labels (int) - label u format discrete/continuous Default: True
    encoder_n_hidden_layers (int) - number of the MLPEncoder's MLP hidden layers. Default: 2
    encoder_hidden_layer_dim (int) - dimension of the MLPEncoder's MLP hidden layers. Default: 128
    encoder_hidden_layer_activation (nn.Module) - activation function applied to the MLPEncoder's MLP hidden layers. Default: nn.Tanh
    decoder_n_gin_blocks (int) - number of GINBlocks in the GINFlowDecoder. Default: 2
    decoder_gin_block_depth (int) - depth of each GINBlock in the GINFlowDecoder. Default: 2
    decoder_affine_input_layer_slice_dim (int) - index at which to split an n-dimensional sample x input to each AffineCouplingLayer. Default: None
    decoder_affine_n_hidden_layers (int) - number of each AffineCouplingLayer's MLP hidden layers. Default: 2
    decoder_affine_hidden_layer_dim (int) - dimension of each AffineCouplingLayer's MLP hidden layers. Default: None
    decoder_affine_hidden_layer_activation (nn.Module) - activation function applied to each AffineCouplingLayer's MLP hidden layers. Default: nn.ReLU
    decoder_nflow_n_hidden_layers (int) - number of the NFlowLayer's MLP hidden layers. Default: 2
    decoder_nflow_hidden_layer_dim (int) - dimension of the NFlowLayer's MLP hidden layers. Default: None
    decoder_nflow_hidden_layer_activation (nn.Module) - activation function applied to the NFlowLayer's MLP hidden layers. Default: nn.ReLU
    decoder_observation_model (str) - GINFlowDecoder's observation model poisson/gaussian Default: "poisson"
    decoder_fr_clamp_min (float) - min value used when clamping decoded firing rates. Default: 1E-7
    decoder_fr_clamp_max (float) - max value used when clamping decoded firing rates. Default: 1E7
    z_prior_n_hidden_layers (int) - number of the ZPriorContinuous's MLP hidden layers. Default: 2
    z_prior_hidden_layer_dim (int) - dimension of the ZPriorContinuous's MLP hidden layers. Default: 32
    z_prior_hidden_layer_activation (nn.Module) - activation function applied to the ZPriorContinuous's MLP hidden layers. Default: nn.Tanh

    Notes
    -----
    decoder_affine_input_layer_slice_dim (int) - when None, x_dim // 2 is assigned. Otherwise assigns the specified value, assuming 0 < decoder_affine_input_layer_slice_dim < x_dim.
    decoder_affine_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(decoder_affine_hidden_layer_dim, x_dim // 4).
    decoder_nflow_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(decoder_nflow_hidden_layer_dim, x_dim // 4).
    """
    
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        z_dim: int,
        discrete_labels: bool = True,
        encoder_n_hidden_layers: int = 2,
        encoder_hidden_layer_dim: int = 128,
        encoder_hidden_layer_activation: nn.Module = nn.Tanh,
        decoder_n_gin_blocks: int = 2,
        decoder_gin_block_depth: int = 2,
        decoder_affine_input_layer_slice_dim: int = None,
        decoder_affine_n_hidden_layers: int = 2,
        decoder_affine_hidden_layer_dim: int = None,
        decoder_affine_hidden_layer_activation: nn.Module = nn.ReLU,
        decoder_nflow_n_hidden_layers: int = 2,
        decoder_nflow_hidden_layer_dim: int = None,
        decoder_nflow_hidden_layer_activation: nn.Module = nn.ReLU,
        decoder_observation_model: str = 'poisson',
        decoder_fr_clamp_min: float = 1E-7,
        decoder_fr_clamp_max: float = 1E7,
        z_prior_n_hidden_layers: int = 2,
        z_prior_hidden_layer_dim: int = 32,
        z_prior_hidden_layer_activation: nn.Module = nn.Tanh
        ) -> None:
        super().__init__()

        if decoder_observation_model == 'gaussian':
            self.observation_noise_model = nn.Linear(
                in_features=1,
                out_features=x_dim, 
                bias=False
            )
        elif decoder_observation_model == 'poisson':
            self.observation_noise_model = None
        else:
            raise ValueError(f"Invalid observation model: {decoder_observation_model}")
        
        self.decoder_observation_model = decoder_observation_model
        
        self.decoder = GINFlowDecoder(
            x_dim=x_dim,
            z_dim=z_dim,
            n_gin_blocks=decoder_n_gin_blocks,
            gin_block_depth=decoder_gin_block_depth,
            affine_input_layer_slice_dim=decoder_affine_input_layer_slice_dim,
            affine_n_hidden_layers=decoder_affine_n_hidden_layers,
            affine_hidden_layer_dim=decoder_affine_hidden_layer_dim,
            affine_hidden_layer_activation=decoder_affine_hidden_layer_activation,
            nflow_n_hidden_layers=decoder_nflow_n_hidden_layers,
            nflow_hidden_layer_dim=decoder_nflow_hidden_layer_dim,
            nflow_hidden_layer_activation=decoder_nflow_hidden_layer_activation,
            observation_model=decoder_observation_model
        )

        if decoder_fr_clamp_min < decoder_fr_clamp_max:
            self.decoder_fr_clamp_min = decoder_fr_clamp_min
            self.decoder_fr_clamp_max = decoder_fr_clamp_max
        else:
            raise ValueError(f"decoder_fr_clamp_min: {decoder_fr_clamp_min} must be less than decoder_fr_clamp_max: {decoder_fr_clamp_max}")
        
        
        self.encoder = MLPEncoder(
            x_dim=x_dim,
            z_dim=z_dim,
            n_hidden_layers=encoder_n_hidden_layers,
            hidden_layer_dim=encoder_hidden_layer_dim,
            activation=encoder_hidden_layer_activation
        )
        
        if discrete_labels:
            self.z_prior = ZPriorDiscrete(
                u_dim=u_dim,
                z_dim=z_dim
            )
        else:
            self.z_prior = ZPriorContinuous(
                u_dim=u_dim,
                z_dim=z_dim,
                n_hidden_layers=z_prior_n_hidden_layers,
                hidden_layer_dim=z_prior_hidden_layer_dim,
                hidden_layer_activation=z_prior_hidden_layer_activation
            )

        self.inference = False

    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None
        ) -> dict[str, torch.Tensor]:

        # Encode each sample observation x to latent z approximating q(z|x)
        encoder_z_mean, encoder_z_log_variance = self.encoder(x)

        with torch.no_grad():
            # Sample latent z using reparameterization trick
            encoder_z_sample = encoder_z_mean + torch.exp(0.5 * encoder_z_log_variance) * torch.randn_like(encoder_z_mean)

            # Generate firing rate using sampled latent z
            encoder_firing_rate = self.decoder(encoder_z_sample)

            if self.decoder_observation_model == 'poisson':
                encoder_firing_rate = torch.clamp(encoder_firing_rate, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)
        
        if self.inference:
            return {
                'encoder_firing_rate': encoder_firing_rate,
                'encoder_z_sample': encoder_z_sample,
                'encoder_z_mean': encoder_z_mean,
                'encoder_z_log_variance': encoder_z_log_variance                
            }
        else:
            # Mean and log of variance for each sample using label prior p(z|u)
            lambda_mean, lambda_log_variance = self.z_prior(u)

            # Compute the full posterior of q(z|x,u)~q(z|x)p(z|u) as a product of Gaussians
            posterior_mean, posterior_log_variance = self.compute_posterior(encoder_z_mean, encoder_z_log_variance, lambda_mean, lambda_log_variance)  

            # Sample latent z using reparameterization trick
            posterior_z_sample = posterior_mean + torch.exp(0.5 * posterior_log_variance) * torch.randn_like(posterior_mean)

            # Generate firing rate using sampled latent z
            posterior_firing_rate = self.decoder(posterior_z_sample)
            if self.decoder_observation_model == 'poisson':
                posterior_firing_rate = torch.clamp(posterior_firing_rate, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)

            return {
                'encoder_firing_rate': encoder_firing_rate,
                'encoder_z_sample': encoder_z_sample,
                'encoder_z_mean':   encoder_z_mean,
                'encoder_z_log_variance': encoder_z_log_variance,
                'lambda_mean': lambda_mean,
                'lambda_log_variance': lambda_log_variance,
                'posterior_firing_rate': posterior_firing_rate,
                'posterior_z_sample': posterior_z_sample,
                'posterior_mean': posterior_mean,
                'posterior_log_variance': posterior_log_variance
            }

    def decode(
        self,
        x: torch.Tensor
        ) -> torch.Tensor:
        """
        Projects samples in the model's latent space (`z_dim`) into the model's observation space (`x_dim`) by passing them through the model's decoder module.

        Parameters
        ----------
        `x` (Tensor) - sample(s) the model's latent space (`z_dim`) to be decoded. `Size([n_samples, z_dim])`

        Returns
        -------
        `decoded` (Tensor) - sample(s) in the model's observation space (`x_dim`). `Size([n_samples, x_dim])` 
        """

        with torch.no_grad():
            decoded = self.decoder(x)

            if self.decoder_observation_model == 'poisson':
                decoded = torch.clamp(decoded, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)

        return decoded

    def encode(
        self,
        x: torch.Tensor,
        return_stats: bool = False
        ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Projects samples in the model's observation space (`x_dim`) into the model's latent space (`z_dim`) by passing them through the model's encoder module.

        Parameters
        ----------
        - `x` (Tensor) - the sample(s) in the model's observation space (`x_dim`) to be encoded. `Size([n_samples, x_dim])`
        - `return_stats` (bool) - if `True`, the mean and log of variance associated with the encoded sample are returned; otherwise only the encoded sample is returned.

        Returns
        -------
        - `encoded` (Tensor) - sample(s) in the model's latent space(`z_dim`). `Size([n_samples, z_dim])`
        - `encoded_mean` (Tensor) [optional] - mean(s) of the encoded sample(s). `Size([n_samples, z_dim])`
        - `encoded_log_variance` (Tensor) [optional] - log of variance(s) of the encoded sample(s). `Size([n_samples, z_dim])`
        """

        with torch.no_grad():
            encoded_mean, encoded_log_variance = self.encoder(x)
            # Sample latent z using reparameterization trick
            encoded = encoded_mean + torch.exp(0.5 * encoded_log_variance) * torch.randn_like(encoded_mean)

        if return_stats:
            return encoded, encoded_mean, encoded_log_variance
        else:
            return encoded

    def get_label_statistics(
        self,
        u: Union[float, int, list, tuple, torch.Tensor],
        device: Optional[torch.device] = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log of the variance associated with label `u`.

        Parameters
        ----------
        - `u` (int, float, list, tuple, or Tensor) - the label of the generated samples
        - `device` (torch.device) - the torch device on which the model resides

        Returns
        -------
        - `label_mean` (Tensor) - mean associated with the specified label. `Size([1, z_dim])`
        - `label_log_variance` (Tensor) - log of the variance associated with  the specified label. `Size([1, z_dim])`
        """

        with torch.no_grad():
            if isinstance(u, int): # discrete label
                u = torch.as_tensor([u], device=device)
            elif isinstance(u, float): # continuous label
                u = torch.as_tensor([u], device=device).unsqueeze(dim=0)
            elif isinstance(u, list) or isinstance(u, tuple): # continuous label
                u = torch.as_tensor(u, dtype=torch.float, device=device).unsqueeze(dim=0)

            # Mean and log of variance of label u using label prior estimator of p(z|u)
            label_mean, label_log_variance = self.z_prior(u) 
        
        return label_mean, label_log_variance
    
    def sample(
        self,
        u: Union[float, int, list, tuple, torch.Tensor],
        n_samples: int = 1,
        device: Optional[torch.device] = None
        ) -> torch.Tensor:
        """
        Generates samples in the model's x dimension using a specified label u.

        Parameters
        ----------
        - u (int, float, list, tuple, or Tensor) - the label of the generated samples
        - n_samples (int) - the number of samples to generate
        - device (torch.device) - the torch device on which the model resides

        Returns
        -------
        samples (Tensor) - the generated samples corresponding to the specified label. Size([n_samples, x_dim]) 
        """
        with torch.no_grad():
            if isinstance(u, int): # discrete label
                u = torch.as_tensor([u], device=device)
            elif isinstance(u, float): # continuous label
                u = torch.as_tensor([u], device=device).unsqueeze(dim=0)
            elif isinstance(u, list) or isinstance(u, tuple): # continuous label
                u = torch.as_tensor(u, dtype=torch.float, device=device).unsqueeze(dim=0)
            
            # Mean and log of variance of label u
            mean, log_variance = self.z_prior(u)

            # Create covariance matrix
            variance = torch.exp(log_variance).squeeze(dim=0)
            covar = torch.eye(mean.size(dim=1), device=device) # N x N

            # Update diagonal with variances
            for idx in range(mean.size(dim=1)):
                covar[idx, idx] = variance[idx].item()

            # Create distribution used for sampling
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covar)
            
            # Generate z samples
            samples = distribution.sample((n_samples,))
            samples = samples.squeeze(dim=1)

            # Lift samples to x dimension
            samples = self.decoder(samples)                                 

        return samples

    def sample_z(
        self,
        u: Union[float, int, list, tuple, torch.Tensor],
        n_samples: int = 1,
        device: Optional[torch.device] = None
        ) -> torch.Tensor:
        """
        Generates samples in the model's z dimension using a specified label u.

        Parameters
        ----------
        - u (int, float, list, tuple, or Tensor) - the label of the generated samples
        - n_samples (int) - the number of samples to generate
        - device (torch.device) - the torch device on which the model resides

        Returns
        -------
        samples (Tensor) - the generated samples corresponding to the specified label. Size([n_samples, z_dim]) 
        """

        with torch.no_grad():
            if isinstance(u, int): # discrete label
                u = torch.as_tensor([u], device=device)
            elif isinstance(u, float): # continuous label
                u = torch.as_tensor([u], device=device).unsqueeze(dim=0)
            elif isinstance(u, list) or isinstance(u, tuple): # continuous label
                u = torch.as_tensor(u, dtype=torch.float, device=device).unsqueeze(dim=0)

            # Mean and log of variance of label u
            mean, log_variance = self.z_prior(u)

            # Create covariance matrix
            variance = torch.exp(log_variance).squeeze(dim=0)
            covar = torch.eye(mean.size(dim=1), device=device) # N x N

            # Update diagonal with variances
            for idx in range(mean.size(dim=1)):
                covar[idx, idx] = variance[idx].item()

            # Create distribution used for sampling
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covar)
                
            # Generate z samples
            samples = distribution.sample((n_samples,))
            samples = samples.squeeze(dim=1)                          

        return samples
    
    def set_inference_mode(
        self,
        state: bool
        ) -> None:
        """
        Toggles the model's inference state flag. When `True`, the model's `forward` method does not utilize the `u` parameter. When `False`,  the `u` parameter is utilized.

        Parameters
        ----------
        `state` (bool) - the desired inference state

        Returns
        -------
        `None`
        """

        self.inference = state

    @staticmethod
    def compute_posterior(
    z_mean: torch.Tensor, 
    z_log_variance: torch.Tensor, 
    lambda_mean: torch.Tensor, 
    lambda_log_variance: torch.Tensor,
    ) -> torch.Tensor:
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
