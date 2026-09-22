# Paper Activity 03: LeNet

**Paper:** LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998).

## Reproduce
Build a small LeNet-style network for MNIST or FashionMNIST:
convolution -> pooling -> convolution -> pooling -> dense classifier.

## Compare
Train a parameter-matched MLP baseline.

Report:
- parameter count
- test accuracy
- training time
- robustness to small translations

## Explain
Why are local receptive fields and weight sharing appropriate inductive biases for images?

## Ablation
Replace convolutions with a dense layer or remove pooling.
