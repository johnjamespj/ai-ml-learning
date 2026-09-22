# Lab 08: VAE Math

Paper companion: Auto-Encoding Variational Bayes.

Implement the reparameterization trick and KL divergence term before building a full VAE.

Run:
```bash
LAB_TARGET=starter pytest labs/lab08_vae_math/test_lab.py -q
```

Then sample many z values and verify their empirical mean/std match the requested latent Gaussian.
