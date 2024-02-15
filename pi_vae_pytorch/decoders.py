import torch
from torch import nn

from pi_vae_pytorch.layers import AffineCouplingBlock, NFlowLayer

class GINFlowDecoder(nn.Module):
    """
    Define mean(p(x|z)) using GIN volume preserving flow.
    nflow_layer > affine_block(n_lsyers) > activation > out
    """
    def __init__(self,
                 x_dim,
                 z_dim,
                 n_affine_blocks,
                 affine_block_depth=2,
                 affine_input_layer_slice_dim=None,
                 affine_n_hidden_layers=2,
                 affine_hidden_layer_dim=30,
                 affine_hidden_layer_activations=['relu', 'relu', 'identity'],
                 nflow_n_hidden_layers=2,
                 nflow_hidden_layer_dim=30,
                 nflow_hidden_layer_activations=['relu', 'relu', 'identity'],
                 observation_model="poisson"):
        super().__init__()
        self.output_activation = nn.Softplus() if observation_model == "poisson" else nn.Identity()

        self.n_flow = NFlowLayer(x_dim=x_dim,
                                 z_dim=z_dim,
                                 n_hidden_layers=nflow_n_hidden_layers,
                                 hidden_layer_dim=nflow_hidden_layer_dim,
                                 hidden_layer_activations=nflow_hidden_layer_activations)

        self.affine_blocks = nn.ModuleList([AffineCouplingBlock(x_dim=x_dim,
                                                                n_affine_layers=affine_block_depth,
                                                                affine_input_layer_slice_dim=affine_input_layer_slice_dim,
                                                                affine_n_hidden_layers=affine_n_hidden_layers,
                                                                affine_hidden_layer_dim=affine_hidden_layer_dim,
                                                                affine_hidden_layer_activations=affine_hidden_layer_activations)
                                            for _ in range(n_affine_blocks)])
        
        self.permutation_idxs = [torch.randperm(x_dim) for i in range(self.n_affine_blocks)]
    
    def forward(self, z):
        # NFlow layer
        output = self.n_flow(z)
        
        # Affine blocks
        for i in range(self.n_affine_blocks):
            # Randomly permute input
            output = torch.gather(output, -1, self.permutation_idxs[i])

            output = self.affine_blocks[i](output)

        # Softplus or Identity
        return self.output_activation(output)
