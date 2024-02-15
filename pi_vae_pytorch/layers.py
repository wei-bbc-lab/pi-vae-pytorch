import torch
from torch import nn


class MLP(nn.Module):
    """
    A basic MLP module.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 n_hidden_layers=2,
                 hidden_layer_dim=30,
                 activations=["tanh", "tanh", "identity"]):
        super().__init__()
        self.net = nn.ModuleList([])

        # Construct layer dims
        dims = [in_features]
        for _ in range(n_hidden_layers * 2):
            dims.append(hidden_layer_dim)
        dims.append(out_features)

        # Build layers
        for i in range(n_hidden_layers+1):
            if activations[i] == "tanh":
                activation = nn.Tanh()
            elif activations[i] == "identity":
                activation = nn.Identity()

            self.net.append(
                nn.ModuleList([
                    nn.Linear(dims[2*i], dims[2*i+1]),
                    activation
                ])
            )

    def forward(self, x):
        return self.net(x)


class AffineCouplingLayer(nn.Module):
    """
    Maps input x to [x_{1:dd}, x_{dd+1:n} * exp(s(x_{1:dd})) + t(x_{1:dd})].
    """
    def __init__(self,
                 x_dim,
                 x_slice_dim=None,
                 n_hidden_layers=2,
                 hidden_layer_dim=30,
                 hidden_layer_activations=['relu', 'relu', 'identity']):
        super().__init__()
        self.net = nn.ModuleList([])
        self.x_dim = x_dim

        if x_slice_dim is None:
            self.x_slice_dim = self.x_dim // 2
        else:
            self.x_slice_dim = x_slice_dim

        # Select hidden dim
        hidden_dim = max(hidden_layer_dim, x_dim // 4)
        # Input layer dim
        dims = [self.x_slice_dim]
        # Hidden layer dims
        for _ in range(n_hidden_layers * 2):
            dims.append(hidden_dim)
        # Output layer dim
        dims.append(2 * (x_dim - x_slice_dim) - 1)

        # Build layers
        for i in range(n_hidden_layers+1):
            if hidden_layer_activations[i] == "relu":
                activation = nn.ReLU()
            elif hidden_layer_activations[i] == "tanh":
                activation = nn.Tanh()
            elif hidden_layer_activations[i] == "identity":
                activation = nn.Identity()

            self.net.append(
                nn.ModuleList([
                    nn.Linear(dims[2*i], dims[2*i+1]),
                    activation
                ])
            )

    def forward(self, x):
        # Split input
        x_1 = torch.narrow(x, 0, 0, self.x_slice_dim)
        x_2 = torch.narrow(x, 0, self.x_slice_dim, self.x_dim - self.x_slice_dim)

        st_out = self.net(x_1)
        s_out = torch.narrow(st_out, 0, 0, self.x_dim - self.x_slice_dim - 1)
        t_out = torch.narrow(st_out, 0, self.x_dim - self.x_slice_dim - 1, self.x_dim - self.x_slice_dim)

        # clamp func to ensure output of s is small
        s_out = 0.1 * nn.Tanh(s_out)
        # enforce the last layer has sum 0
        s_out = torch.cat((s_out, torch.sum(torch.neg(s_out), keepdim=True)), dim=-1)

        # perform transformation
        transform_x = x_2 * torch.exp(s_out) + t_out

        return torch.cat((transform_x, x_1), dim=-1)


class AffineCouplingBlock(nn.Module):
    """
    A block of AffineCouplingLayers. 
    """
    def __init__(self,
                 x_dim,
                 n_affine_layers=2,
                 affine_input_layer_slice_dim=None,
                 affine_n_hidden_layers=2,
                 affine_hidden_layer_dim=30,
                 affine_hidden_layer_activations=['relu', 'relu', 'identity']):
        super().__init__()
        self.net = nn.ModuleList(
            [
                AffineCouplingLayer(x_dim=x_dim,
                                    x_slice_dim=affine_input_layer_slice_dim,
                                    n_hidden_layers=affine_n_hidden_layers,
                                    hidden_layer_dim=affine_hidden_layer_dim,
                                    hidden_layer_activations=affine_hidden_layer_activations)
            ] * n_affine_layers
        )

    def forward(self, x):
        return self.net(x)


class NFlowLayer(nn.Module):
    """
    Define the first layer in GIN flow, which maps z to the cancatenation of z and t(z), t is parameterized by self.net. 
    This is equivalent to GIN model with input as z1:dim_z padding dim_x - dim_z zeros.
    """
    def __init__(self,
                 x_dim,
                 z_dim,
                 n_hidden_layers=2,
                 hidden_layer_dim=30,
                 hidden_layer_activations=['relu', 'relu', 'identity']):
        super().__init__()
        self.net = nn.ModuleList([])

        # Compute used hidden dim
        hidden_dim = max(hidden_layer_dim, x_dim // 4)
        # Input dim
        dims = [z_dim]
        # Hidden layer dims
        for _ in range(n_hidden_layers * 2):
            dims.append(hidden_dim)
        # Output layer
        dims.append(x_dim - z_dim)

        # Build layers
        for i in range(n_hidden_layers+1):
            if hidden_layer_activations[i] == "relu":
                activation = nn.ReLU()
            elif hidden_layer_activations[i] == "tanh":
                activation = nn.Tanh()
            elif hidden_layer_activations[i] == "identity":
                activation = nn.Identity()

            self.net.append(
                nn.ModuleList([
                    nn.Linear(dims[2*i], dims[2*i+1]),
                    activation
                ])
            )

    def forward(self, z):
        """
        Maps z to the cancatenation of z and t(z). 
        """
        t_z = self.net(z)
        return torch.cat((z, t_z), dim=-1)
