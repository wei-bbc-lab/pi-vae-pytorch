# Poisson Identifiable VAE (pi-VAE)

This is a Pytorch implementation of [Poisson Identifiable VAE (pi-VAE)](https://arxiv.org/abs/2011.04798), used to construct latent variable models of neural activity while simultaneously modeling the relation between the latent and task variables (non-neural variables, e.g. sensory, motor, and other externally observable states).

The original implementation by [Dr. Ding Zhou](https://zhd96.github.io/) and [Dr. Xue-Xin Wei](https://sites.google.com/view/xxweineuraltheory/) in Tensorflow 1.13 is available [here](https://github.com/zhd96/pi-vae).

Another Pytorch implementation by [Dr. Lyndon Duong](https://github.com/lyndond) is available [here](https://github.com/lyndond/lyndond.github.io/blob/0865902edb4648a8690ed8d449573d9236a72406/code/2021-11-25-pivae.ipynb).

## Install

```
pip install pi-vae-pytorch
```

## Usage

```
import torch
from pi_vae_pytorch import PiVAE

model = PiVAE(
      x_dim = 1,
      u_dim = 1,
      z_dim = 1
)
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
