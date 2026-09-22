# Paper Activity 06: Variational Autoencoder

**Paper:** Kingma & Welling, "Auto-Encoding Variational Bayes" (2013), https://arxiv.org/abs/1312.6114

## Reproduce
1. Train a normal autoencoder.
2. Train a VAE.
3. Plot latent samples.
4. Interpolate between two latent points.
5. Sample from the prior and decode.

## Key mechanism
Implement the reparameterization:
z = mu + sigma * epsilon

## Ablation
Remove or drastically reduce the KL term.

## Explain
Why is the reparameterization trick important for gradient-based learning?
