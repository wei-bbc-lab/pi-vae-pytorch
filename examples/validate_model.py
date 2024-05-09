# This is a refactoring of Lyndon Duong's pi-VAE validation code which can be used to validate pi-vae-pytorch. His original work is available here:
# https://github.com/lyndond/lyndond.github.io/blob/0865902edb4648a8690ed8d449573d9236a72406/code/2021-11-25-pivae.ipynb

import torch
import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Tuple
import seaborn as sns
from pytorch_model_summary import summary
from pi_vae_pytorch import PiVAE
from pi_vae_pytorch.utils import compute_loss


#############################################
######          CONFIGURATION          ######
#############################################

seed = 69420 # Pytorch manual seed

discrete_prior = True
n_samples = 10000 # number of data samples to generate
n_classes = 5 # number of classes (discrete only)
x_dim = 100 # dimension of observations

n_epochs = 1000 # number of training epochs
valid_every = 10 # run validation every _ epochs
learning_rate = 5E-4

plots_dir = "/an/example/path/" # path to save directory for training plots 

#############################################


class MLP(nn.Module):
    """ Multilayer Perceptron (MLP) with n layers and fixed activation after each hidden
    layer, with linear output layer.

    Parameters
    ----------
    in_channels: Number of channels in input.
    out_channels: Number of output channels.
    n_layers: Number of layers including output layer.
    hidden_size: Size of each hidden layer.
    activation: Activation after each hidden layer.
    """

    def __init__(self,
        in_channels: int, 
        out_channels: int, 
        n_layers: int, 
        hidden_size: int = 32, 
        activation: nn.Module = nn.ReLU,
    ):
        assert n_layers >= 3
        super().__init__()

        layers = [nn.Linear(in_channels, hidden_size), activation()]  # first layer
        for _ in range(n_layers-2):  # intermediate layers
            layers.extend( [nn.Linear(hidden_size, hidden_size), activation()])
        layers.extend([nn.Linear(hidden_size, out_channels)])  # last layer
    
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class Permutation(nn.Module):
    """ A permutation layer.

    Parameters
    ----------
    in_channels: Number of channels to be permuted.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer('p', torch.randperm(in_channels))
        self.register_buffer('invp', torch.argsort(self.p))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_channels
        return x[:, self.p]

    def backward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_channels
        return x[:, self.invp]


class RealNVPLayer(nn.Module):
    """ Single layer of Real non-volume preserving (NVP) transform.
    Uses a multilayer perceptron (MLP) to transform half input channels into scaling and
    translation vectors to be used on remaining half.

    Parameters
    ----------
    in_channels: Number of channels of input.
    n_layers: Number of layers in MLP.
    activation: Activation of first (n_layers-1) layers (default ReLU).
    """

    def __init__(
        self, 
        in_channels: int, 
        n_layers: int = 3, 
        activation: nn.Module = nn.ReLU,
        ):
        super().__init__()
        assert in_channels%2 == 0, "Should have even dims"

        self.mlp = MLP(
            in_channels=in_channels//2, 
            out_channels=in_channels, 
            n_layers=n_layers, 
            hidden_size=in_channels//2,
            activation=activation,
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """Splits x into two halves (x0, x1), maps x0 through MLP to form s and t, 
        then returns (s*x1+t, x0).
        """
        # split
        x0, x1 = torch.chunk(x, chunks=2, dim=-1)

        st = self.mlp(x0)

        # scale and translate
        s, t = torch.chunk(st, chunks=2, dim=-1)
        s =  .1 * torch.tanh(s)  # squash s
        transformed = x1 * torch.exp(s) + t
        y = torch.cat([transformed, x0], axis=-1)
        return y


class RealNVPBlock(nn.Module):
    """Real Non-volume preserving (NVP) block, consisting of n_layers layers.

    Parameters
    ----------
    in_channels: Number of channels of input.
    n_layers: Number of layers in each block.
    """

    def __init__(self, in_channels: int, n_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.nvp = nn.Sequential(
            *[RealNVPLayer(in_channels) for _ in range(n_layers)]
            )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.nvp(x)


class RealNVP(nn.Module):
    """ Real (non-volume preserving) NVP model (forward only).

    Parameters
    ----------
    in_channels: Size of NVP input. Half will remain untouched.
    n_blocks: Number of NVP blocks.
    n_layers: Number of layers within each NVP block.
    """

    def __init__(self, in_channels: int, n_blocks: int, n_layers: int):
        super().__init__()
        blocks = [RealNVPBlock(in_channels, n_layers)]

        for _ in range(n_blocks-1):
            blocks.extend([
                           Permutation(in_channels), 
                           RealNVPBlock(in_channels, n_layers)
                           ]
            )

        self.nvp = nn.Sequential(*blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        """Run n_blocks of NVP"""
        return self.nvp(x)


def lift(
    x: Tensor, 
    out_channels: int, 
    nvp_blocks: int, 
    nvp_layers: int
    ) -> Tensor:
    """ Nonlinearly transform x from lo -> hi dim w/ random-initialized MLP and realNVP.
    MLP lifts from in_channels to out_channels, then realNVP further transforms the 
    data.
    
    Parameters
    ----------
    x: Input Tensor with Size([n, in_channels])
    out_channels: Dimensionality of output.
    nvp_blocks: Number of NVP blocks in NVP model.
    nvp_layers: Number of layers within each NVP block.

    Returns
    -------
    y: Output Tensor with size Size([n, out_channels]).
    """

    in_channels = x.shape[-1]
    mlp = MLP(in_channels, out_channels-in_channels, n_layers=3, activation=nn.ReLU).requires_grad_(False)
    realnvp_model = RealNVP(out_channels, nvp_blocks, nvp_layers).requires_grad_(False)
    
    # fill ambient space
    x_append = mlp(x)  
    y = torch.cat([x, x_append], -1)
    y = realnvp_model(y)  
    return y


def simulate_data_discrete(
    n_samples: int, 
    n_cls: int, 
    n_dim: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """ Create n-dimensional data with 2D intrinsic dimensionality.

    Parameters
    ----------
    n_samples: Number of samples.
    n_cls: Number of discrete classes.
    n_dim: Dimensionality of data.

    Returns
    -------
    x_true: Data to be used, Size([n_samples, n_dim]).
    z_true: True 2D data, Size([n_samples, 2]).
    u_true: Condition labels with n_cls unique classes, Size([n_samples, ]).
    lam_true: Poisson rate parameter of each sample, Size([n_samples, n_dim]).
    """

    mu_true = torch.empty((2, n_cls)).uniform_(-5, 5)
    var_true = torch.empty((2, n_cls)).uniform_(.5, 3)
    u_true = torch.tile(torch.arange(n_cls), (n_samples//n_cls, ))

    z0 = torch.normal(mu_true[0][u_true], np.sqrt(var_true[0][u_true]))
    z1 = torch.normal(mu_true[1][u_true], np.sqrt(var_true[1][u_true]))
    z_true = torch.stack([z0, z1], -1)

    ## Nonlinearly lift from 2D up to n_dims using RealNVP
    mean_true = lift(z_true, n_dim, nvp_blocks=4, nvp_layers=2)
    lam_true = torch.exp(2*torch.tanh(mean_true))  # Poisson rate param
    x_true = torch.poisson(lam_true)
    return x_true, z_true, u_true, lam_true


def simulate_data_continuous(n_samples: int, n_dim: int) -> torch.Tensor:
    """TODO: docstring. Haven't validated yet."""
    ## true 2D latent

    u_true = torch.empty((n_samples, )).uniform_(0, 2*np.pi)
    mu_true = torch.stack([u_true, 2*torch.sin(u_true)], -1)
    var_true = .15 * torch.abs(mu_true)
    var_true[:,0] = .6 - var_true[:,1]

    z_true = torch.randn((n_samples, 2)) * torch.sqrt(var_true) + mu_true

    mean_true = lift(z_true, n_dim, nvp_blocks=4, nvp_layers=2)
    lam_true = torch.exp(2.2 * torch.tanh(mean_true))
    x_true = torch.poisson(lam_true)

    return x_true, z_true, u_true, lam_true


"""
Pytorch setup
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

torch.manual_seed(seed)


"""
Generate sim data
"""

if discrete_prior:
    
    x_true, z_true, u_true, lam_true = simulate_data_discrete(n_samples, n_classes, x_dim)
else:
    x_true, z_true, u_true, lam_true = simulate_data_continuous(n_samples, x_dim)


"""
Format sim data
"""

x_true = x_true.to(device)
u_true = u_true.to(device)

x_all = x_true.reshape(50, -1, x_dim)
u_all = u_true.reshape(50, -1)
u_all = u_all if discrete_prior else u_all.unsqueeze(-1)

x_train = x_all[:40]
u_train = u_all[:40]

x_valid = x_all[40:45]
u_valid = u_all[40:45]

x_test = x_all[45:]
u_test = u_all[45:]


"""
Initialize model
"""

vae = PiVAE(
    x_dim=x_dim,
    u_dim=n_classes if discrete_prior else 1,
    z_dim=2,
    discrete_labels=discrete_prior,
    encoder_n_hidden_layers=3,
    encoder_hidden_layer_dim=60,
    decoder_affine_n_hidden_layers=3,
    decoder_affine_hidden_layer_dim=30,
    decoder_nflow_n_hidden_layers=3,
    decoder_nflow_hidden_layer_dim=30,
    decoder_observation_model="poisson"
).to(device)

if discrete_prior:
    print(summary(vae, torch.rand(200,x_dim,device=device), torch.randint(0, n_classes, (200,), device=device)))
else:
    print(summary(vae, x_all[0], u_all[0]))

optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate)


"""
Train model
"""

loss_train = []
loss_valid = []
n_valid = len(x_valid)
n_train = len(x_train)
n_samples = x_train[0].shape[0]
pbar = tqdm(range(n_epochs))

for epoch in pbar:
    train_loss = 0.

    for batch in range(n_train):
        optimizer.zero_grad()
        x, u = x_train[batch], u_train[batch]
        outputs = vae(x, u)
        loss = compute_loss(
            x=x,
            firing_rate=outputs["firing_rate"],
            lambda_mean=outputs["lambda_mean"],
            lambda_log_variance=outputs["lambda_log_variance"],
            posterior_mean=outputs["posterior_mean"],
            posterior_log_variance=outputs["posterior_log_variance"],
            observation_model="poisson"
        )
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / n_train
    
    loss_train.append(train_loss) 

    if epoch % valid_every == 0:
        with torch.no_grad():
            valid_loss = 0

            for i in range(n_valid):
                x, u = x_valid[i], u_valid[i]
                outputs = vae(x, u)
                loss = compute_loss(
                    x=x,
                    firing_rate=outputs["firing_rate"],
                    lambda_mean=outputs["lambda_mean"],
                    lambda_log_variance=outputs["lambda_log_variance"],
                    posterior_mean=outputs["posterior_mean"],
                    posterior_log_variance=outputs["posterior_log_variance"],
                    observation_model="poisson"
                )
                valid_loss += loss.item() / n_valid
            
            if np.isnan(loss):
                print("Loss is nan")
                break

            pbar.set_postfix({"valid-loss": f"{valid_loss:.04E}"})
            loss_valid.append(valid_loss)


"""
Plot results
"""

with sns.plotting_context("talk"):
    fig, ax = plt.subplots(1, 1)
    ax.plot(loss_train, ".-", label="train")
    ax.plot(np.arange(1, len(loss_valid)+1) * valid_every, loss_valid, ".-", label="valid")
    ax.set(xlabel="epoch", ylabel="neg ELBO")
    ax.legend()
    sns.despine()
    fig.savefig(f"{plots_dir}loss.png")

with torch.no_grad():
    outputs = vae(x_true, u_true if discrete_prior else u_true.unsqueeze(-1))
    post_means = outputs["posterior_mean"]
    post_log_vars = outputs["posterior_log_variance"]
    post_lam_means = outputs["lambda_mean"]
    post_lam_log_vars = outputs["lambda_log_variance"]
    z_means = outputs["z_mean"]
    z_log_vars = outputs["z_log_variance"]

post_means = post_means.cpu()
z_means = z_means.cpu()
z_true = z_true.cpu()
u_true = u_true.cpu()
ll = n_samples

if discrete_prior:
    c_vec = np.array(['crimson','orange','dodgerblue','limegreen','indigo'])
    idx = u_true
else:
    length = 30
    c_vec = plt.cm.viridis(np.linspace(0,1, length))
    bins = np.linspace(0, 2*np.pi, length)
    centers = (bins[1:]+bins[:-1])/2
    idx = np.digitize(u_true.squeeze(), centers)

with sns.plotting_context("talk", font_scale=1):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    ax[0].scatter(z_true[:,0], z_true[:,1], c=c_vec[idx], s=1, alpha=0.5, rasterized=True)
    ax[1].scatter(post_means[:,1], post_means[:,0], c=c_vec[idx], s=1, alpha=0.5, rasterized=True)
    ax[2].scatter(z_means[:,1], z_means[:,0], s=1, c=c_vec[idx], alpha=0.5, rasterized=True)

    ax[0].set(xlabel="latent 1", ylabel="latent 2", title="ground truth")
    ax[1].set(xlabel="latent 1", ylabel="latent 2", title=r"posterior means $q(z|x,u)\propto q(z|x)p(z|u)$")
    ax[2].set(xlabel="latent 1", ylabel="latent 2", title=r"encoder mean $q(z|x)$")

    fig.tight_layout()
    sns.despine()
    fig.savefig(f"{plots_dir}latents.png")
