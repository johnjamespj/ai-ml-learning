# Lesson 46: Autoencoders and VAEs

## Autoencoder idea
An autoencoder learns to compress an input into a latent representation and reconstruct the original input.

Pipeline:

x -> encoder -> z -> decoder -> x_hat

The simplest objective is reconstruction loss.

## PyTorch skeleton
```python
import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, d_in=784, d_latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.ReLU(),
            nn.Linear(256, d_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
```

## Latent space
The latent vector is a compressed learned representation. Inspect it using PCA or simple scatter plots.

## Variational autoencoders
A VAE learns a distribution over latent variables rather than one deterministic code.

Key ideas:
- mean and log variance
- reparameterization trick
- reconstruction term
- KL-divergence regularization

## Exercise
Train an autoencoder on MNIST/FashionMNIST. Compare reconstruction quality as latent dimension changes. Then inspect whether neighboring latent vectors produce similar decoded outputs.

## Signal connection
Autoencoders can learn compact representations of spectra or spectrograms and may be useful for denoising and anomaly detection.
