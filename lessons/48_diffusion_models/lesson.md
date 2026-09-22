# Lesson 48: Diffusion models

## Core idea
Diffusion models learn to reverse a gradual noising process.

Forward process:
clean sample -> progressively noisier samples

Learned reverse process:
noise -> progressively denoised sample

## Training intuition
A common formulation:
1. sample clean data x0;
2. sample a timestep t;
3. add known Gaussian noise;
4. ask a neural network to predict the noise;
5. optimize prediction error.

## Important concepts
- noise schedule
- timesteps
- denoising network
- score intuition
- latent diffusion
- conditional generation
- classifier-free guidance

## Exercise
Start in 1-D or 2-D. Train a tiny network to denoise samples from a simple distribution across noise levels.

## Why this matters
The same broad idea underlies many modern image-generation systems, but the training objective is much easier to understand on toy data.
