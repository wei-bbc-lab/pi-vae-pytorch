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

x = torch.randn(1, 100) # Size([n_samples, x_dim])

u = torch.randn(1, 3) # Size([n_samples, u_dim])

outputs = model(x, u) # dict
```

### Parameters

- `x_dim`: int  
    Dimension of observation `x`
- `u_dim`: int  
    Dimension of label `u`
- `z_dim`: int  
    Dimension of latent `z`
- `discrete_labels`: bool  
    - Default: `True`  

    Flag denoting `u`'s label type - `True`: discrete or `False`: continuous.
- `encoder_n_hidden_layers`: int  
    - Default: `2`  

    Number of hidden layers in the MLP of the model's encoder. 
- `encoder_hidden_layer_dim`: int  
    - Default: `120`  

    Dimensionality of each hidden layer in the MLP of the model's encoder. 
- `encoder_hidden_layer_activation`: nn.Module    
    - Default: `nn.Tanh`  

    Activation function applied to the outputs of each hidden layer in the MLP of the model's encoder. 
- `decoder_n_gin_blocks`: int  
    - Default: `2`  

    Number of GIN blocks used within the model's decoder. 
- `decoder_gin_block_depth`: int   
    - Default: `2`  

    Number of AffineCouplingLayers which comprise each GIN block.
- `decoder_affine_input_layer_slice_dim`: int  
    - Default None (corresponds to `x_dim / 2`)  

    Index at which to split an n-dimensional input x. 
- `decoder_affine_n_hidden_layers`: int  
    - Default: `2`  

    Number of hidden layers in the MLP of the model's encoder. 
- `decoder_affine_hidden_layer_dim`: int  
    - Default: `None` (corresponds to `x_dim / 4`)  

    Dimensionality of each hidden layer in the MLP of each AffineCouplingLayer. 
- `decoder_affine_hidden_layer_activation`: nn.Module  
    - Default: `nn.ReLU`  

    Activation function applied to the outputs of each hidden layer in the MLP of each AffineCouplingLayer. 
- `decoder_nflow_n_hidden_layers`: int  
    - Default: `2`  

    Number of hidden layers in the MLP of the decoder's NFlowLayer. 
- `decoder_nflow_hidden_layer_dim`: int  
    - Default: `None` (corresponds to `x_dim / 4`)  

    Dimensionality of each hidden layer in the MLP of the decoder's NFlowLayer. 
- `decoder_nflow_hidden_layer_activation`: nn.Module   
    - Default: `nn.ReLU`  

    Activation function applied to the outputs of each hidden layer in the MLP of the decoder's NFlowLayer. 
- `decoder_observation_model`: str  
    - Default: `poisson`  
    - One of `gaussian` or `poisson`

    Observation model used by the model's decoder. 
- `decoder_fr_clamp_min`: float  
    - Default: `1E-7`  
    - Only applied when `decoder_observation_model="poisson"`

    Mininimum threshold used when clamping decoded firing rates.
- `decoder_fr_clamp_max`: float  
    - Default: `1E7` 
    - Only applied when `decoder_observation_model="poisson"`

    Maximum threshold used when clamping decoded firing rates.
- `z_prior_n_hidden_layers`: int  
    - Default: `2`  
    - Only applied when `discrete_labels=False`  

    Number of hidden layers in the MLP of the ZPriorContinuous module. 
- `z_prior_hidden_layer_dim`: int  
    - Default: `20`  
    - Only applied when `discrete_labels=False`

    Dimensionality of each hidden layer in the MLP of the ZPriorContinuous module. 
- `z_prior_hidden_layer_activation`: nn.Module  
    - Default: `nn.Tanh`  
    - Only applied when `discrete_labels=False`

    Activation function applied to the outputs of each hidden layer in the MLP of the decoder's ZPriorContinuous module. 

### Returns

A dicitonary with the following items. 

- `firing_rate`: Tensor   
    - Size([n_samples, x_dim])  

    Predicted firing rates of `z_sample`. 
- `lambda_mean`: Tensor  
    - Size([n_samples, z_dim])  

    Mean for each sample using label prior p(z \| u). 
- `lambda_log_variance`: Tensor  
    - Size([n_samples, z_dim])  
    
    Log of variance for each sample using label prior p(z \| u). 
- `posterior_mean`: Tensor  
    - Size([n_samples, z_dim])  

    Mean for each sample using full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u). 
- `posterior_log_variance`: Tensor  
    - Size([n_samples, z_dim])  

    Log of variance for each sample using full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u). 
- `z_mean`: Tensor  
    - Size([n_samples, z_dim])  

    Mean for each sample using approximation of q(z \| x). 
- `z_log_variance`: Tensor  
    - Size([n_samples, z_dim])  

    Log of variance for each sample using approximation of q(z \| x). 
- `z_sample`: Tensor  
    - Size([n_samples, z_dim])  
    
    Generated latents `z`. 

## Loss Function

### Poisson observation model

```
from pi_vae_pytorch.utils import compute_loss

outputs = model(x, u) # Initialized with decoder_observation_model="poisson" 

loss = compute_loss(
    x=x,
    firing_rate=outputs["firing_rate"],
    lambda_mean=outputs["lambda_mean"],
    lambda_log_variance=outputs["lambda_log_variance"],
    posterior_mean=outputs["posterior_mean"],
    posterior_log_variance=outputs["posterior_log_variance"],
    observation_model=model.decoder_observation_model
)

loss.backward()
```

### Gaussian observation model

```
from pi_vae_pytorch.utils import compute_loss

outputs = model(x, u) # Initialized with decoder_observation_model="gaussian" 

loss = compute_loss(
    x=x,
    firing_rate=outputs["firing_rate"],
    lambda_mean=outputs["lambda_mean"],
    lambda_log_variance=outputs["lambda_log_variance"],
    posterior_mean=outputs["posterior_mean"],
    posterior_log_variance=outputs["posterior_log_variance"],
    observation_model=model.decoder_observation_model,
    observation_noise_model=model.observation_noise_model
)

loss.backward()
```

### Parameters

- `x`: Tensor  
    - Size([n_samples, x_dim])  

    Observations `x`.  
- `firing_rate`: Tensor 
    - Size([n_samples, x_dim])  

    Predicted firing rate of generated latent `z`. 
- `lambda_mean`: Tensor 
    - Size([n_samples, z_dim])  
    
    Means from label prior p(z \| u). 
- `lambda_log_variance`: Tensor 
    - Size([n_samples, z_dim])  
    
    Log of variances from label prior p(z \| u). 
- `posterior_mean`: Tensor 
    - Size([n_samples. z_dim])  
    
    Means from full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u). 
- `posterior_log_variance`: Tensor 
    - Size([n_samples. z_dim])  
    
    Log of variances from full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u).
- `observation_model`: str  
    - One of `poisson` or `gaussian`  
    - Should use the same value passed to `decoder_observation_model` when initializing `PiVAE`.  

    The observation model used by pi-VAE's decoder.
- `observation_noise_model`: nn.Module 
    - Default: None  
    - Only applied when `observation model="gaussian"`  
    
    The noise model used when pi-VAE's decoder utilizes a Gaussian observation model. When `PiVAE` is initialized with `decoder_observation_model="gaussian"`, the model's `observation_noise_model` attribute can be used.

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
