# Lesson 47: Generative adversarial networks

A GAN has two competing models:

- Generator: produces synthetic examples
- Discriminator: tries to distinguish real from generated examples

The generator improves by learning to fool the discriminator.

## Concept
The discriminator sees:
- real x -> should output real
- G(z) -> should output fake

The generator learns from discriminator feedback.

## Main difficulties
- unstable training
- mode collapse
- sensitivity to architecture and hyperparameters
- difficult evaluation

## Exercise
Train a tiny GAN on a simple 2-D distribution before attempting images. Plot generated samples during training.

## Important
GANs are powerful historically and conceptually, but many modern image-generation workflows rely heavily on diffusion models instead.
