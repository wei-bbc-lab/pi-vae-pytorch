from typing import Dict

from torch import clamp, nn, Tensor

from pi_vae_pytorch.decoders import GINFlowDecoder
from pi_vae_pytorch.encoders import MLPEncoder
from pi_vae_pytorch.layers import ZPriorContinuous, ZPriorDiscrete
from pi_vae_pytorch.utils import compute_posterior, generate_latent_z


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
    encoder_hidden_layer_dim (int) - dimension of the MLPEncoder's MLP hidden layers. Default: 120
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
    z_prior_hidden_layer_dim (int) - dimension of the ZPriorContinuous's MLP hidden layers. Default: 20
    z_prior_hidden_layer_activation (nn.Module) - activation function applied to the ZPriorContinuous's MLP hidden layers. Default: nn.Tanh

    Notes
    -----
    decoder_affine_input_layer_slice_dim (int) - when None, x_dim // 2 is assigned. Otherwise assigns the specified value, assuming 0 < decoder_affine_input_layer_slice_dim < x_dim.
    decoder_affine_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(affine_hidden_layer_dim, x_dim // 4).
    decoder_nflow_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(nflow_hidden_layer_dim, x_dim // 4).
    """
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        z_dim: int,
        discrete_labels: bool = True,
        encoder_n_hidden_layers: int = 2,
        encoder_hidden_layer_dim: int = 120,
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
        decoder_observation_model: str = "poisson",
        decoder_fr_clamp_min: float = 1E-7,
        decoder_fr_clamp_max: float = 1E7,
        z_prior_n_hidden_layers: int = 2,
        z_prior_hidden_layer_dim: int = 20,
        z_prior_hidden_layer_activation: nn.Module = nn.Tanh
        ) -> None:
        super().__init__()

        if decoder_observation_model == "gaussian":
            self.observation_noise_model = nn.Linear(
                in_features=1,
                out_features=x_dim, 
                bias=False
            )
        elif decoder_observation_model == "poisson":
            self.observation_noise_model = None
        else:
            raise ValueError(f"Invalid observation model: {decoder_observation_model}")
        
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

        self.decoder_fr_clamp_min = decoder_fr_clamp_min
        self.decoder_fr_clamp_max = decoder_fr_clamp_max
        self.decoder_observation_model = decoder_observation_model
        
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

    def forward(
        self,
        x: Tensor,
        u: Tensor
        ) -> Dict[str, Tensor]:
        
        # Mean and log of variance for each sample using label prior p(z|u)
        lambda_mean, lambda_log_variance = self.z_prior(u) 

        # Map each sample observation x to latent z approximating q(z|x)
        z_mean, z_log_variance = self.encoder(x)  

        # Compute the full posterior of q(z|x,u)~q(z|x)p(z|u) as a product of Gaussians
        posterior_mean, posterior_log_variance = compute_posterior(z_mean, z_log_variance, lambda_mean, lambda_log_variance)  

        # Sample latent z using reparameterization trick
        z_sample = generate_latent_z(
            mean=posterior_mean,
            log_variance=posterior_log_variance
        )  

        # Generate firing rate using sampled latent z
        firing_rate = self.decoder(z_sample)
        if self.decoder_observation_model == "poisson":
            firing_rate = clamp(firing_rate, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)

        return {
            "firing_rate": firing_rate,
            "lambda_mean": lambda_mean,
            "lambda_log_variance": lambda_log_variance,
            "posterior_mean": posterior_mean,
            "posterior_log_variance": posterior_log_variance,
            "z_mean": z_mean,
            "z_log_variance": z_log_variance,
            "z_sample": z_sample
        }
