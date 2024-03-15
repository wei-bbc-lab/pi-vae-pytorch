# Poisson Identifiable VAE (pi-VAE)

This is a Pytorch implementation of [Poisson Identifiable VAE (pi-VAE)](https://arxiv.org/abs/2011.04798), used to construct latent variable models of neural activity while simultaneously modeling the relation between the latent and task variables (non-neural variables, e.g. sensory, motor, and other externally observable states).

The original implementation by [Dr. Ding Zhou](https://zhd96.github.io/) and [Dr. Xue-Xin Wei](https://sites.google.com/view/xxweineuraltheory/) in Tensorflow 1.13 is available [here](https://github.com/zhd96/pi-vae).

Another Pytorch implementation by [Dr. Lyndon Duong](http://lyndonduong.com/) is available [here](https://github.com/lyndond/lyndond.github.io/blob/0865902edb4648a8690ed8d449573d9236a72406/code/2021-11-25-pivae.ipynb).

## Install

```
pip install pi-vae-pytorch
```

## Usage

```
import torch
from pi_vae_pytorch import PiVAE

model = PiVAE(
    x_dim = 100,
    u_dim = 3,
    z_dim = 2,
    discrete_labels=False
)

# Size([n_samples, x_dim])
x = torch.randn(1, 100) 

# Size([n_samples, u_dim])
u = torch.randn(1, 3) 

outputs = model(x, u)
```

## Parameters

- `x_dim`: int  
    Dimension of observation `x`
- `u_dim`: int  
    Dimension of label `u`
- `z_dim`: int  
    Dimension of latent `z`
- `discrete_labels`: bool  
    Flag denoting `u`'s label type - `True`/discrete or `False`/continuous. Default: `True`
- `encoder_n_hidden_layers`: int  
    Number of hidden layers in the MLP of the model's encoder. Default: `2`
- `encoder_hidden_layer_dim`: int  
    Dimensionality of each hidden layer in the MLP of the model's encoder. Default: `120`
- `encoder_hidden_layer_activation`: nn.Module    
    Activation function applied to the outputs of each hidden layer in the MLP of the model's encoder. Default: `nn.Tanh`
- `decoder_n_gin_blocks`: int  
    Number of GIN blocks used within the model's decoder. Default: `2`
- `decoder_gin_block_depth`: int   
    Number of AffineCouplingLayers which comprise each GIN block
- `decoder_affine_input_layer_slice_dim`: int  
    Index at which to split an n-dimensional input x. Default None (corresponds to `x_dim / 2`)
- `decoder_affine_n_hidden_layers`: int  
    Number of hidden layers in the MLP of the model's encoder. Default: `2`
- `decoder_affine_hidden_layer_dim`: int  
    Dimensionality of each hidden layer in the MLP of each AffineCouplingLayer. Default: `None` (corresponds to `x_dim / 4`)
- `decoder_affine_hidden_layer_activation`: nn.Module  
    Activation function applied to the outputs of each hidden layer in the MLP of each AffineCouplingLayer. Default: `nn.ReLU`
- `decoder_nflow_n_hidden_layers`: int  
    Number of hidden layers in the MLP of the decoder's NFlowLayer. Default: `2`
- `decoder_nflow_hidden_layer_dim`: int  
    Dimensionality of each hidden layer in the MLP of the decoder's NFlowLayer. Default: `None` (corresponds to `x_dim / 4`)
- `decoder_nflow_hidden_layer_activation`: nn.Module = nn.ReLU,  
    Activation function applied to the outputs of each hidden layer in the MLP of the decoder's NFlowLayer. Default: `nn.ReLU`
- `decoder_obervation_model`: str  
    Observation model used by the model's decoder (`gaussian` | `poisson`). Default: `poisson`
- `z_prior_n_hidden_layers`: int  
    Number of hidden layers in the MLP of the ZPriorContinuous module. Default: `2`
- `z_prior_hidden_layer_dim`: int  
    Dimensionality of each hidden layer in the MLP of the ZPriorContinuous module. Default: `20`
- `z_prior_hidden_layer_activation`: nn.Module  
    Activation function applied to the outputs of each hidden layer in the MLP of the decoder's ZPriorContinuous module. Default: `nn.Tanh`

## Outputs

- `firing_rate`: predicted firing rates of `z_sample`. Size([n_samples, x_dim])
- `lambda_mean`: Size([n_samples, z_dim])
- `lambda_log_variance`: Size([n_samples, z_dim])
- `posterior_mean`: Size([n_samples, z_dim])
- `posterior_log_variance`: Size([n_samples, z_dim])
- `z_mean`: Size([n_samples, z_dim])
- `z_log_variance`: Size([n_samples, z_dim])
- `z_sample`: generated latents. Size([n_samples, z_dim])

## Loss Function

```
from pi_vae_pytorch.utils import compute_loss

outputs = model(x, u)

loss = compute_loss(
    x=x,
    firing_rate=outputs["firing_rate"],
    lambda_mean=outputs["lambda_mean"],
    lambda_log_variance=outputs["lambda_log_variance"],
    posterior_mean=outputs["posterior_mean"],
    posterior_log_variance=outputs["posterior_log_variance"],
    observation_model="poisson",
    observation_noise_model=nn.Linear(
        in_features=1,
        out_features=x_dim, 
        bias=False
    )
)

loss.backward()
```

## Citation

```
@misc{zhou2020learning,
    title={Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE}, 
    author={Ding Zhou and Xue-Xin Wei},
    year={2020},
    eprint={2011.04798},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
