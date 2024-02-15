from torch import nn
from pi_vae_pytorch.decoders import GINFlowDecoder
from pi_vae_pytorch.layers import MLP


class PiVAE(nn.Module):
    """
    The pi-VAE model.
    """
    def __init__(self,
                 x_dim,
                 u_dim,
                 z_dim,
                 discrete_labels=True,
                 encoder_n_hidden_layers=2,
                 encoder_hidden_layer_dim=30,
                 encoder_hidden_layer_activations=["tanh", "tanh", "identity"],
                 decoder_n_affine_blocks=2,
                 decoder_affine_block_depth=2,
                 decoder_affine_input_layer_slice_dim=None,
                 decoder_affine_n_hidden_layers=2,
                 decoder_affine_hidden_layer_dim=30,
                 decoder_affine_hidden_layer_activations=["relu", "relu", "identity"],
                 decoder_nflow_n_hidden_layers=2,
                 decoder_nflow_hidden_layer_dim=30,
                 decoder_nflow_hidden_layer_activations=["relu", "relu', 'identity"],
                 decoder_obervation_model="poisson"):
            super().__init__()

            self.encoder = MLP(in_features=x_dim,
                               out_features=encoder_hidden_layer_dim,
                               n_hidden_layers=encoder_n_hidden_layers,
                               hidden_layer_dim=encoder_hidden_layer_dim,
                               activations=encoder_hidden_layer_activations)
            
            self.decoder = GINFlowDecoder(x_dim=x_dim,
                                          z_dim=z_dim,
                                          n_affine_blocks=decoder_n_affine_blocks,
                                          affine_block_depth=decoder_affine_block_depth,
                                          affine_input_layer_slice_dim=decoder_affine_input_layer_slice_dim,
                                          affine_n_hidden_layers=decoder_affine_n_hidden_layers,
                                          affine_hidden_layer_dim=decoder_affine_hidden_layer_dim,
                                          affine_hidden_layer_activations=decoder_affine_hidden_layer_activations,
                                          nflow_n_hidden_layers=decoder_nflow_n_hidden_layers,
                                          nflow_hidden_layer_dim=decoder_nflow_hidden_layer_dim,
                                          nflow_hidden_layer_activations=decoder_nflow_hidden_layer_activations,
                                          observation_model=decoder_obervation_model)

    def forward(self,x):
        pass